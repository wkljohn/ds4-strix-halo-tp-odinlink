# Prep research for the TP FFN gate-overlap redesign (not yet implemented)

Status: **on hold for Codex** (see `PREFILL-PROFILE.md`'s "CONFIRMED WIN"
section and the remaining ~50%+ estimated un-overlapped gate cost). This doc
is external research done while Codex was unavailable
(quota resets ~2026-08-11), to have concrete design options ready rather
than starting the redesign from a blank page whenever it resumes. Nothing
here has been implemented or validated on our hardware - treat every claim
below as "reported by a paper/source," not "confirmed on ds4."

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
