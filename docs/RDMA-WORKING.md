# OdinLink RDMA working under ds4 TP=2

    ds4-tp: rdma device odl_tb5_0 (port state 4)
    ds4: tp rdma: provider rejected UC, using RC
    ds4-tp: worker connected, transport=rdma
    tensor parallelism bound: rank 0, 50/50 expert split, rdma transport

Correct output ("The answer is Paris"), no dropped completions, no CQ-full
messages on either rank.

## A/B at identical settings (-n 100, --temp 0, same prompt)

| transport | prefill | generation | correct |
|---|---|---|---|
| TCP | 11.85 t/s | 8.29 t/s | yes |
| RDMA | 10.79 t/s | **8.57 t/s** | yes |

**Decode +3.4%, prefill -9%.** Both differences are small and the sample is 100
tokens, so treat this as "RDMA is roughly at parity, slightly ahead on decode".
+3.4% matches an independent estimate of ~5% derived from the gate arithmetic
(86 gates x ~85us saved RTT against a ~150ms token).

CAUTION on prefill numbers: the same TCP configuration measured 32.15 t/s at
-n 200 and 11.85 t/s at -n 100. Prefill t/s here is dominated by a short fixed
prompt and is not comparable across run lengths. An earlier note in this repo
claiming "RDMA makes prefill 3x worse" compared mismatched runs and was wrong.

## What it took

1. **CQ ring sized from `cqe`** (patches/odinlink/cq-dynamic-ring.patch). The
   ring was `ibv_wc ring[64]` regardless of the requested depth, while
   `base.cqe` echoed back whatever was asked - the provider advertised a depth
   it did not have. ds4 requests 512 and a bulk round can post 65 completions
   (64 recv + 1 signalled send); `odl_cq_post` drops on overflow and the
   consumer then waits forever. Now allocated from `cqe+1`, floored at the old
   64, capped at 65536, freed on destroy, and `base.cqe`/`max_cqe` report the
   truth.
   Built to `build-ds4/`, deployed to `~/odl-ds4/` on both nodes. **The system
   shim used by the working vLLM/RCCL stack was left untouched.**

2. **The peer was missing `libodl_tb5.so.0` entirely.** The head has it
   installed system-wide; the peer did not, so the verbs shim failed to
   `dlopen` there. This presented as ds4's generic "no active device", which is
   why it looked like a probe/hardware problem. Shipping the matching core
   library next to the shim fixed it - both nodes now enumerate `odl_tb5_0`
   with port state 4 (ACTIVE) and an IPv4-mapped GID.

3. **`DS4_TP_VERBS_LIB`** points ds4 at the right shim, bypassing the loader's
   search order (it tries `libodl_tb5_verbs.so` first, and a stale copy of that
   name exists under models/odl-verbs/).

4. **Patch 8 (QP type fallback) earned its keep**: "provider rejected UC, using
   RC". ds4 asks for UC; OdinLink implements RC only.

## Run recipe

    DS4_TP_VERBS_LIB=$HOME/odl-ds4/libodl_tb5_verbs.so.0.1.0 \
    LD_LIBRARY_PATH=$HOME/odl-ds4 \
    DS4_TP_BIG_DIRECT=1 \
    ./ds4 -m <Q4_K.gguf> --rocm --tensor-parallel --role {coordinator|worker} \
          {--listen 10.4.0.1 5599 | --coordinator 10.4.0.1 5599} \
          --transport rdma -c 4096 --temp 0

Start the worker first.

**UPDATE (2026-08-06): the Q8-to-F16 cache is now disabled by default.** It is
available only with `DS4_ROCM_ENABLE_Q8_F16_CACHE=1`, which prints a warning
because it consumes about 10 GiB per rank and can drive reported VRAM usage
near 99%. Ordinary inference and tests must leave that switch unset.
It was believed required because the cache takes ~9.9 GiB and the MoE arena
OOMs without it - re-tested directly and this is no longer true (or never
was, under today's code): the cache has its own graceful budget cap
(`cuda_q8_f16_cache_has_budget`) that stops at ~9.85-9.91 GiB and falls back
cleanly, always leaving ~4.8 GiB free. Leaving the cache enabled recovers
output_proj's fast cuBLAS-f16 path (silently dead this whole session under
the old flag) and measured **94-99 t/s prefill, up from 74.80-75.65 t/s** -
validated correct via an exact `--dump-logprobs` diff (0/30 greedy tokens
differ) and safe over a 500-token decode run (no OOM on either rank). See
`PREFILL-PROFILE.md`'s "CONFIRMED WIN" section and
[[ds4-q8f16-cache-prefill-win-2026-08-05]] for full detail. If a future
model/config genuinely needs the old flag again, re-derive why before
reaching for it - don't copy it forward as boilerplate.
