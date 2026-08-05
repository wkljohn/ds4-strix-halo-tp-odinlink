# Prep research for the TP FFN gate-overlap redesign (not yet implemented)

Status: **designed, not implemented.** Original external-paper research
below was done while Codex was quota-exhausted; once quota reset, Codex
itself read the real ds4 code and produced a concrete, ds4-specific design
(see "Codex's concrete design" section below) - that supersedes the external
papers as the primary plan. Nothing has been implemented or validated on our
hardware yet.

## Codex's concrete design (2026-08-05, after reading the real code)

Dispatched Codex directly at this codebase once quota reset. Key findings
(each grounded in an exact file:line Codex read, not inferred):

**The gate is genuinely, fully blocking today - confirmed by reading the
actual code, not assumed:** `ds4_gpu_tp_big_gate_encode()` queues an arrival
marker then a `hipStreamWaitValue64` (`ds4_rocm.cu:714`) on a deliberately
blocking stream (`g_tp_stream`); the service thread performs the RDMA
exchange synchronously and only releases the GPU after (`ds4_tp.c:1210`,
OdinLink waits for every send/receive completion before returning). The
canonical rank-ordered sum (`ds4.c:29133`) and HC expansion (`ds4.c:29160`)
both wait behind the full exchange. There is no gate/compute overlap in the
current FFN path, full stop.

**"Overlap with next-layer attention" (my first instinct) is INVALID -
ruled out by Codex, not just deprioritized:** the next layer's attention
reads `batch_cur_hc`, which is swapped in only after this layer's HC
expansion completes (`ds4.c:29238`) - so next-layer Q/K/V/attention
transitively depends on THIS layer's FFN all-reduce. Starting it early would
feed it an incomplete residual state and change the model's output. This is
also why Ladder-Residual (candidate technique 1 below) needs a different
residual topology, not just a scheduling change - confirms that paper's
caveat was justified.

**Recommended approach instead: row-chunk pipelining of the SAME layer's
gate exchange with its own HC residual expansion** (not FLUX's full
kernel-fusion - Codex judged that unnecessary for a first version):
1. Opt-in `DS4_TP_FFN_GATE_OVERLAP=1`, default-off, restricted to ROCm TP
   prefill with `DS4_TP_BIG_DIRECT=1`, no directional FFN steering.
2. Split the exchange matrix into row-aligned chunks (128 or 256 rows).
3. Build persistent row views of `batch_routed_out`, `batch_ffn_out`/direct
   receive slab, `batch_after_attn_hc`, `batch_next_hc`, shared-output
   buffers.
4. Queue ALL chunk exchanges up front without an immediate stream wait.
5. Per chunk, in sequence: wait for that chunk's release, do the canonical
   rank-ordered add for only those rows, run `hc_expand_*` for those rows.
6. While the GPU handles chunk N's add+HC-expand, the service thread
   exchanges chunk N+1 - this is the actual overlap.
7. Do not swap `batch_next_hc` into the next layer until every chunk is done.

Concrete touch points: factor `ds4_tp_encode()` into separate
reserve/publish and wait operations near `ds4_rocm.cu:714`; implement the
already-DECLARED-but-stubbed `ds4_gpu_tp_big_gate_kick()` /
`ds4_gpu_tp_big_gate_wait()` APIs (`ds4_gpu.h:287`) for real; add the gated
chunk path around `ds4.c:29122`; reuse row views with the HC kernels at
`rocm/ds4_rocm_hc_output_launch.cuh:258`.

**Correctness hazards Codex flagged explicitly** (read before implementing):
row-view objects must stay alive until the service thread finishes
dereferencing them (freeing after `kick` instead of after the matching
`wait` is a use-after-free - the EXACT bug class that caused the row-split
crash, see [[ds4-attn-rowsplit-crash-2026-08-05]]); a peer receive chunk
must not be read before its release word is observed; the rank-ordered add
must stay rank-0-then-rank-1 for every chunk (reversing operands on either
rank introduces floating-point divergence); shared-expert rows must be
folded exactly once (`tp_row_split_ffn` has different addend semantics from
the ordinary and `shared_down_f16` paths); both ranks must compute identical
chunk counts/sizes or the sequence/header protocol desyncs.

**Codex did NOT implement this** - judged it too risky to write blind
without the ROCm split-gate primitive already existing, and it isn't. Left
as a design for careful, incremental implementation.

**Required hardware validation when implemented** (Codex's own list): build
+ `make test-tp-hello`; run overlap-off vs `DS4_TP_FFN_GATE_OVERLAP=1` with
identical prompt/seed/context/chunk-size/`--temp 0`; require 0 selected-token
differences via `--dump-logprobs`, byte-identical text, no missing/duplicate
steps, no TP sequence/header/RDMA/timeout errors; check per-step logits for
near-exact (not just "close enough") parity since chunking preserves
canonical operand order; benchmark 3+ long-prefill samples only after
correctness passes; sweep prompt token counts below/at/above/not-divisible-by
the chunk size; use `DS4_TP_BIGGATE_PROFILE=1` (see `PREFILL-PROFILE.md`'s
"Clean, direct FFN gate-cost measurement" section - this flag gives a real
~39% gate-cost baseline to compare against, found in this same research
pass) to quantify how much overlap is actually achieved, separately from
end-to-end t/s.

## External paper research (done earlier, while Codex was quota-exhausted)

## Candidate technique 1: Ladder-Residual (arXiv 2501.06589)

"Parallelism-aware architecture for accelerating large model inference with
communication overlapping." Restructures the ORDER of residual-stream
dependencies so attention/MLP compute for a later stage can proceed while a
prior stage's all-reduce is still in flight - a "ladder" pattern that breaks
the strict sequential dependency between a collective and the compute that
normally waits on it.

**Caveat, not yet resolved:** an automated summary of this paper claimed it
needs no retraining and preserves exact computational equivalence - but the
paper's own title calls it an "architecture," and reordering residual-stream
dependencies typically changes what each layer's pretrained weights actually
see as input. **Do not trust "no retraining needed" without reading the full
paper directly** (not just an automated summary) before considering this for
ds4, since we cannot retrain DeepSeek-V4-Flash. This may turn out to require
a model trained with this residual topology from the start, which would make
it inapplicable to us.

## Candidate technique 2: FLUX (arXiv 2406.06858) - more promising, pure systems technique

"Fast Software-based Communication Overlap On GPUs Through Kernel Fusion."
Over-decomposes both the communication op (e.g. all-reduce) and the
dependent compute op into fine-grained pieces, then FUSES them into a single
larger GPU kernel so the GPU can interleave communication and compute at the
kernel-scheduling level rather than waiting on a full collective before
starting the next op.

- **No model retraining required** - pure inference-engine/kernel technique,
  applies to an already-trained model as-is.
- Demonstrated on NVLink-connected GPUs, "across various GPU generations and
  interconnects" - some hardware specificity, but the core idea (fine-grained
  decompose + fuse) is conceptually interconnect-agnostic; our OdinLink RDMA
  transport would need its own low-level primitive to actually interleave at
  this granularity (NVSHMEM-equivalent), which does not obviously exist for
  OdinLink today - this is the biggest open unknown for porting the idea.
- **Reported results**: up to 96% communication overlap; up to 1.66x faster
  PREFILL than vLLM; up to 1.30x faster decode; up to 1.24x training speedup
  on 128 GPUs.
- Engineering complexity: fine-grained kernel decomposition + fusion is
  real, non-trivial systems work - a bigger lift than a config change, in
  the same ballpark of effort as this session's row-split attempt
  ([[ds4-attn-rowsplit-crash-2026-08-05]], the one that crashed) - meaning
  it should get the SAME rigor: mandatory `--dump-logprobs` differential
  validation, not a coherence smoke test, before ever being considered safe
  to ship.

## Candidate technique 3 (background, not a new lever): Megatron-LM's column-then-row MLP split

Classical technique: split the MLP's first (up-projection) matrix
COLUMN-wise and the second (down-projection) matrix ROW-wise across TP
ranks, so partial sums only need to be combined ONCE per MLP block (a single
all-reduce), instead of needing a sync after every sub-op. ds4's current
design already appears to do something equivalent (one FFN gate exchange
per layer, not multiple - confirmed by this session's profiling showing a
single `hc_post`/gate boundary per layer, not several). Worth a deliberate
double-check when the gate-overlap work resumes, but likely already correct
here - flagging so nobody re-derives this from scratch and finds a
false-positive "opportunity."

## Recommended next step when resuming this work

1. Read the Ladder-Residual paper in full (not an automated summary) to
   resolve whether it's inference-only or needs retraining - if the latter,
   drop it as inapplicable.
2. Investigate FLUX's actual kernel-fusion mechanics in more depth (the full
   paper, not just the abstract) and assess what OdinLink-side primitive
   would be needed to interleave communication and compute at that
   granularity over Thunderbolt RDMA - this is likely the hard part, harder
   than the GPU-kernel side.
3. Whichever direction, this should go through the same process as every
   other risky change this session: Codex implements/reviews, Sonnet does
   hardware validation with exact logprob diffs, given this project's
   history of silent-correctness bugs surviving smoke tests.
