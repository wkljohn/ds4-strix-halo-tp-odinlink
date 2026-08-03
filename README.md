# ds4 tensor parallelism on AMD Strix Halo (gfx1151)

Runs [antirez/ds4](https://github.com/antirez/ds4) ("DwarfStar") with **two-node
tensor parallelism on ROCm**, over Thunderbolt — either TCP or OdinLink
RDMA. Upstream ds4 refuses tensor parallelism on any backend except Metal; this
tree implements the ROCm side.

**Status: working.** DeepSeek-V4-Flash-0731-abliterated Q4_K (153 GiB) across two
Strix Halo nodes, correct greedy output, **~10.6 t/s decode** over RDMA.

    $ ./ds4 -m DeepSeek-V4-Flash-Q4_K-0731.gguf --rocm --tensor-parallel \
            --role coordinator --listen 10.4.0.1 5599 --transport rdma \
            -c 4096 --temp 0 -p "What is the capital of France? Answer with one word."

    ds4-tp: rdma device odl_tb5_0 (port state 4)
    ds4: tp rdma: provider rejected UC, using RC
    tensor parallelism bound: rank 0, 50/50 expert split, rdma transport
    ... The answer is Paris. ...
    ds4: generation: 11.09 t/s

## Why this exists

The model is 153 GiB and each node has a 96 GiB VRAM carve-out. It does not fit
on one node, and ds4's SSD streaming is incompatible with the generic Q4_K
routed-MoE path (fails at the first MoE layer). Two-node TP is the only way to
run it — and unlike llama.cpp's `-ts`, which is *pipeline* parallelism where
bandwidth is not additive, real TP runs both nodes concurrently.

| | bytes/node/token | measured | own ceiling | efficiency |
|---|---|---|---|---|
| llama.cpp no draft (pipeline) | 16.9 GiB | 9.42 t/s | 13.2 t/s | 71% |
| **this tree (real TP)** | 8.91 GiB | **~10.6 t/s** | 25.1 t/s | 42% |

## Applying

    git clone https://github.com/antirez/ds4 && cd ds4
    git checkout 54b36ed
    git apply /path/to/patches/ds4-strix-halo-tp-complete.patch
    make strix-halo

`patches/patch_ds4_gfx1151_tp.py` applies the first ten changes individually with
fail-closed anchors and `--check`; the consolidated patch is the whole tree.

## Running

Start the **worker first**, then the coordinator.

    # worker (peer)
    DS4_CUDA_NO_Q8_F16_CACHE=1 ./ds4 -m <model.gguf> --rocm --tensor-parallel \
        --role worker --coordinator 10.4.0.1 5599 --transport tcp -c 4096

    # coordinator (head)
    DS4_CUDA_NO_Q8_F16_CACHE=1 ./ds4 -m <model.gguf> --rocm --tensor-parallel \
        --role coordinator --listen 10.4.0.1 5599 --transport tcp -c 4096 \
        --temp 0 -p "..."

Two environment variables are **required**, not tuning knobs:

- `DS4_CUDA_NO_Q8_F16_CACHE=1` — without it the q8->f16 accelerator cache takes
  ~9.9 GiB and the MoE arena OOMs mid-prefill.
- `DS4_TP_PREFILL_SPLIT_MIN=999999` — disables the TP prefill row split, which
  cannot work on ROCm yet (it needs
  `ds4_gpu_attention_prefill_raw_heads_range_tensor`, an unimplemented kernel).
  **Without this, every prompt of 32 or more tokens fails outright.** Decode is
  unaffected. See docs/LONG-PROMPT-BUG.md.

Verified with a 3306-token prompt: prefill 29.18 t/s, generation 10.49 t/s, and
the model correctly comprehends the long context.

For RDMA add `--transport rdma` and point ds4 at an OdinLink verbs shim:

    DS4_TP_VERBS_LIB=$HOME/odl-ds4/libodl_tb5_verbs.so.0.1.0 \
    LD_LIBRARY_PATH=$HOME/odl-ds4

Both nodes need `libodl_tb5.so.0` alongside the shim. See docs/RDMA-WORKING.md.

## What the patches do

| # | change |
|---|---|
| 1-2 | build the verbs path on Linux, not just macOS; dlopen `libibverbs.so.1` |
| 3-4 | accept the ROCm backend for TP; compile the TP engine off-Apple |
| 5 | the ROCm TP gate runtime (two channels, HIP stream wait-value) |
| 6 | define `DS4_ROCM_TP_READY` build-wide |
| 7 | QP type UC -> RC fallback (OdinLink implements RC only) |
| 10 | K-sliced Q8_0 matmul — needs no new kernel, see the header comment |
| 11 | `attention_output_q8_tp` and `hc_expand_add` |
| 12 | expert sharding by **range** (residency: 80.76 GiB/rank) |
| 13 | gate off the **null stream**; dedicated remap scratch buffer |
| 14 | generic dense-quant K-slice (the TP shared-expert split) |
| 16 | fold the routed-MoE addend **after** the launch |
| 17 | service-thread back-off and dead-yield-guard fix (~+23%) |

## The two bugs worth knowing about

**`hipStreamWriteValue64` does not work on the null stream** on gfx1151/ROCm 7.2.
`scripts/t4_null_stream_gate_probe.cpp` isolates it: identical code, nothing
queued, one pair — a created stream sees the arrival in 0.00 s, stream 0 never
does. The gate now uses a dedicated stream, with `t5_*` verifying that legacy
null-stream implicit sync still orders compute behind the gate.

**The routed-MoE addend must be folded after the launch, not before.** Every
terminal write in the ROCm MoE is an assignment (`out[row] = total`), never an
accumulation, so a pre-added addend is overwritten and the shared expert
silently vanishes from every decode layer. Upstream refused this case outright
("Metal-only") because Metal folds it *inside* the down kernel.

## Docs

`docs/` carries the full record, including the wrong turns:
ROOFLINE-PROVENANCE.md (where every number comes from, and three corrections),
NULL-STREAM-ROOT-CAUSE.md, CORRUPTION-BISECT.md, TP-PATH-TRACE.md,
EXPERT-SHARD-DESIGN-FLAW.md, RDMA-WORKING.md, PATCH17-SPIN.md.

`scripts/t*_*.cpp` are standalone probes — each isolates one hardware question
and can be re-run on a new ROCm or new silicon.

## Not done

- Attention head split (ceiling 25.1 -> ~36 t/s) — DS4's group slice is
  decode-only and untested; the GLM head split is a different mechanism.
- DSpark/MTP speculative decoding — blocked on
  `ds4_gpu_attention_noncausal_raw_batch_heads_tensor`, an unavailable stub.
- We are at 42% of our own bandwidth ceiling; the gap is not the transport.
