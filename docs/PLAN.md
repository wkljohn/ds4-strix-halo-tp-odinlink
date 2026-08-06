# Plan: ds4 tensor parallelism on 2x AMD Strix Halo over OdinLink RDMA

**Target:** DeepSeek V4 Flash Q4_K served
TP=2 across two gfx1151 nodes, DSpark speculative decoding on, RCCL-free
point-to-point RDMA over Thunderbolt 5 (OdinLink).

**Upstream pinned:** `antirez/ds4` @ `54b36ed` (2026-07-28).
**Strategy:** patches applied to a *stock* upstream tree. Never a fork. Same
discipline as `vllm-strix-dsv4`: idempotent, fail-closed, `--check` dry-run,
`.orig-ds4tp` backups, every patch anchored on a unique string that fails loudly
if upstream moves.

---

## Why this can work at all (established, cited)

| fact | evidence |
|---|---|
| DeepSeek-V4 is the **baseline** TP case, not an excluded arch | `ds4_tp.c:834-836` — "DS4 fires every slot in order (identity mapping); GLM's schedule … skips dense layers" |
| **DSpark works under TP** | `ds4_tp.c:510-512` — drafting allowed on leader, verify mirrored via `DS4_TP_FRAME_VERIFY` |
| **No collectives** — point-to-point per-token gate exchange through a registered GPU slab | `ds4_tp.h`; three variants `gate_exchange` / `batch_gate_exchange` / `big_gate_exchange` |
| RDMA is **standard libibverbs**, not Apple APIs | `ds4_tp.c` dlsyms `ibv_reg_mr`/`ibv_post_send`/`ibv_poll_cq`/… |
| OdinLink **verbs provider already built** | `models/odl-verbs/libodl_tb5-rdmav34.so.34` + `odl_tb5.driver` |

Contrast llama.cpp, which hard-excludes `LLM_ARCH_DEEPSEEK4` from `-sm tensor`
and throws at load. ds4 was built around this model.

## Verdict: GO for a staged proof-of-concept (revised 2026-08-03)

An initial review returned NO-GO on the premise that GPU<->RDMA memory ordering
on gfx1151 was unproven. **That premise is false and our own deployment
disproves it**: vLLM TP=2 ran across both nodes over this transport with
`Using network ODL_TB5` on both ranks, serving DeepSeek-V4-Flash through
answer-key 100/100 and needle 12/12. Cross-node tensor parallelism on this
hardware is demonstrated, not speculative.

Re-reviewed against the OdinLink source, the verdict is **GO for a staged PoC,
5-9 engineer-days**, with the caveat - which we adopt - that no throughput claim
is made before measurement.

### The design fact that shapes everything: HOST-STAGED, not zero-copy

    rccl/src/odl_tb5_net_v7.c:182   props->ptrSupport    = NCCL_PTR_HOST;
    rccl/src/odl_tb5_net_v7.c:188   props->netDeviceType = NCCL_NET_DEVICE_HOST;
    rccl/src/odl_tb5_net_v7.c:237   if (type != NCCL_PTR_HOST) return ncclInternalError;
    rccl/src/odl_tb5_net_v7.c:246   dma-buf reg -> ncclInternalError  (stub)
    rccl/src/odl_tb5_net_v7.c:258   memcpy(tx, data, size); odl_tb5_send(...)

OdinLink registers **host memory only** and stages with a memcpy each way. The
production-proven path therefore never registers GPU memory - so "gfx1151
peer-memory registration" is not a prerequisite, it is simply **not used**.

**On this APU that is cheap.** Strix Halo is UMA: host memory and GPU memory are
the same physical LPDDR5X. A host-visible slab is GPU-accessible without a PCIe
round trip, which is what makes host staging viable here where it would be
costly on a discrete GPU.

Zero-copy remains a LATER option, not a blocker: dma-buf ioctls exist
(`lib/src/odl_tb5_xfer.c:92`), the verbs provider has a dma-buf MR object
(`verbs/src/odl_tb5_verbs_mr.c:60`) and the QP selects dma-buf sends by key
(`verbs/src/odl_tb5_verbs_qp.c:172`). The RCCL plugin just never uses them.

### Integration surface: OdinLink VERBS + a host-visible slab

ds4 already posts ordinary verbs operations (`ds4_tp.c:743`, `:945`), so the
verbs provider is the better-shaped interface than the NCCL-net plugin (which is
a collective-library ABI, host-staged, limited to one receive at
`rccl/src/odl_tb5_net_v7.c:284`).

## The blocker

`ds4_rocm.cu:135` — *"Tensor-parallel gates are Metal-only; stubs keep shared
graph code linkable (TP option validation rejects non-Metal backends)."*

| backend | "Metal-only" stubs | real `ds4_gpu_tp_*` |
|---|---|---|
| Metal (1.79 MB) | 0 | 46 |
| CUDA (1.19 MB) | **0** | **10 — real implementations** |
| **ROCm (6 KB)** | **5** | none |

The comment is **stale**: CUDA implements TP. So the job is HIP-ifying CUDA
kernels into an already gfx11-aware shim (`ds4_rocm.h:114-116` knows
`v_dot4_i32_i8`), not writing a backend from scratch.

---

## Phases, ordered by GPU cost. HARD PAUSE before anything needing >5 GB VRAM.

### PHASE A — zero GPU. Repo, patches, compile.

**A1. Repo + patcher skeleton.** `patches/patch_ds4_gfx1151_tp.py` with
`--check`, idempotence, `.orig-ds4tp` backups, fail-closed anchors.
*Verify:* `--check` on a stock tree reports N pending, 0 applied; twice in a row
is a no-op.

**A2. Patch 1 — Linux verbs enablement.** Two edits in `ds4_tp.c`:
- `#if defined(__APPLE__) && __has_include(<infiniband/verbs.h>)` → also accept
  Linux with `<infiniband/verbs.h>` present.
- `dlopen("/usr/lib/librdma.dylib")` → try `libibverbs.so.1` first on Linux.
*Verify:* compiles; `DS4_TP_HAVE_VERBS` defined on Linux. No GPU.

**A3. Patch 2 — ROCm TP kernels.** REVISED after auditing what ROCm already
has. The job is **much smaller than "6 kernels from scratch"**: ROCm already
implements the base primitives, and two of the TP entry points are pure C
composition over them.

Already REAL in `rocm/*.cuh` — reuse, do not rewrite:

| symbol | status |
|---|---|
| `ds4_gpu_matmul_q8_0_tensor` | real |
| `ds4_gpu_attention_output_low_q8_tensor` | real |
| `ds4_gpu_matmul_q8_0_hc_expand_tensor` | real |

Stubs in `ds4_rocm.cu:134-216` that need work, with the actual effort:

| symbol | what it really needs |
|---|---|
| `ds4_gpu_matmul_q8_0_kslice_tensor` | add `k_off`/`k_cnt` slice params to the **existing** `matmul_q8_0` kernel |
| `ds4_gpu_matmul_q8_0_kslice_rows_tensor` | same, "rows" variant (missing in ROCm) |
| `ds4_gpu_attention_output_low_q8_rows_exact_tensor` | row-subset variant of an existing kernel |
| `ds4_gpu_attention_output_q8_tp_tensor` | **pure C wrapper, no kernel** — `ds4_cuda.cu:15775-15831` just validates, slices `heads`, then calls `attention_output_low_q8_tensor` + `matmul_q8_0_kslice_rows_tensor` |
| `ds4_gpu_hc_expand_add_tensor` | relates to the existing `matmul_q8_0_hc_expand_tensor` |
| `ds4_gpu_tp_gate_encode` / `_batch_` / `_big_` | **the genuinely new part** — encoder close/commit + slab publish, Metal/CUDA semantics translated to HIP streams |

Plus trivial no-op setters (`tp_set_batch_exchange`, `tp_suspend_expert_sharding`,
`tp_keepalive_pause`, `tp_set_attn_head_split`, `tp_set_big_exchange`,
`model_residency_skip`).
*Verify:* **`hipcc` compiles for `--offload-arch=gfx1151` with no GPU present.**
This is the bulk of the work and it is entirely offline.

**A4. Patch 3 — TP backend validation.** Wherever TP option validation "rejects
non-Metal backends", allow ROCm. *Verify:* `--help`/option parse accepts the TP
flags on a ROCm build.

**A5. Verbs bring-up — no GPU, no model.** Make one verbs device enumerate over
OdinLink:
- register the provider in `/etc/libibverbs.d/` with a **host** path (the shipped
  `odl_tb5.driver` points at `/models/odl-verbs/…`, a *container* path)
- resolve why `/sys/class/infiniband/` is empty
- **use the provider, NOT the 42-symbol shim** — the shim lacks
  `ibv_get_device_list` and `ibv_query_gid`, both of which `tp_rdma_probe()` calls
*Verify:* `ibv_devices` lists a device; `ibv_devinfo` shows an active port; a
two-node `ibv_rc_pingpong` (or equivalent) completes over 10.4.0.1↔10.4.0.2.
**This is the highest-uncertainty item and it costs zero VRAM.**

**A6. Standalone TP transport test.** ds4's TP handshake + gate exchange with
*synthetic* buffers, no model. Leader/worker on the two nodes.
*Verify:* handshake completes, gate exchange round-trips, RDMA path selected
(not the TCP fallback), and measure per-exchange latency.

### PHASE B — small GPU, under 5 GB. No pause needed.

**B1. Kernel unit tests.** Each ported HIP kernel against its CUDA/Metal
reference on small synthetic tensors. Budget: a few hundred MB.
*Focus:* `ds4_gpu_attention_output_q8_tp_tensor` (group-boundary head splitting).

**ds4 already FAILS CLOSED here, unlike llama.cpp.** `ds4_cuda.cu:15800`:

    const uint64_t k_off = (uint64_t)group0 * rank;
    const uint64_t k_cnt = (uint64_t)group_cnt * rank;
    if ((k_off % 32u) != 0 || (k_cnt % 32u) != 0) return 0;

It validates slice alignment and **refuses** rather than silently corrupting.
(The 32 is the Q8_0 block size, not a warp width - so it is not a wave64
assumption and needs no change for wave32.) **The ROCm port must preserve this
guard verbatim**; dropping it is exactly how silent corruption gets
reintroduced.

**B2. Tiny-model end-to-end.** Smallest GGUF that loads, TP=2 across nodes.
Proves the whole chain without touching the 146 GB model.

### PHASE C — >5 GB VRAM. **PAUSE AND ASK BEFORE STARTING.**

**C1.** Load `DeepSeek-V4-Flash-Q4-mxfp4-0731.gguf` (146 GB → ~73 GB/node) TP=2.
**C2.** Correctness gates before any speed number: answer-key + needle
retrieval, both baseline-free.
**C3.** DSpark on; measure acceptance rate, then throughput.
**C4.** Compare against the current llama.cpp 2-node pipeline baseline (9-16 t/s,
content-dependent).

---

## Model choice

| file | size | per node @ TP=2 |
|---|---|---|
| `DeepSeek-V4-Flash-Q4-mxfp4-0731.gguf` | **146 GB** | **~73 GB** ← preferred, MXFP4 matches what ds4 expects for DS4 |
| `DeepSeek-V4-Flash-Q4_K-0731.gguf` | 154 GB | ~77 GB |

Both fit two 96 GiB carves. Neither fits one.

## Risks, ranked

1. **Verbs device never enumerates (A5).** Highest uncertainty, zero VRAM. Do it
   first — if OdinLink cannot present a verbs device, the RDMA path is dead and
   we fall back to ds4's TCP transport, which still works but gives up the
   latency win.
2. **wave32 correctness of the grouped attention kernel (B1).** Silent
   corruption that passes smoke tests. Mitigated by unit tests + answer-key/needle
   gates, all of which need no reference stack.
3. **Draft-model compatibility.** MTP acceptance must be measured against the
   exact trunk checkpoint; TP remains useful if acceptance is too low.
4. **ROCm one-GPU-per-process** (`ds4_rocm_compat.cu:21`). Not a problem — one
   GPU per node is exactly our topology.

## What is deliberately NOT in scope

- Forking ds4. Everything is a patch against `54b36ed`.
- Touching the live llama.cpp deployment.
- Any CPU/GPU frequency change (reduction/reallocation only).
