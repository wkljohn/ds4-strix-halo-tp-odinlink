# Patch 8's weight-masking approach is incompatible with TP residency

Found 2026-08-03 by tracing a VRAM anomaly the user flagged as "not normal".
They were right, and chasing a smaller checkpoint would have hidden it.

## Symptom

VRAM climbs to 95.5 of 96 GiB and prefill dies:

    ds4: ROCm model arena alloc failed for moe_gate (1152.00 MiB request): out of memory
    ds4: ROCm end commands failed: an illegal memory access was encountered

## Measurement

Sampling `rocm-smi` every 2 s through a run, against the log stage:

| t | VRAM | stage |
|---|---|---|
| 18 s | **81.16 GiB** | shard warm done - correct, matches the 80.76 GiB prediction |
| 46 s | 81.16 GiB | idle, TP waiting for worker |
| 60 s | **89.08 GiB** | second "loading model tensors into device cache" (+7.9) |
| 62 s | **95.52 GiB** | (+6.4) then arena alloc fails |

The model is fully resident by t=18. A *second* phase then adds ~14.4 GiB.

## Root cause - and it is ours, not upstream's

The failing size is exact:

    145.12 GiB routed experts / 43 layers / 3 tensors (gate|up|down)
      = 1.1250 GiB = 1152 MiB

That is **one layer's FULL expert tensor, all 256 experts**. This rank has only
its half resident (576 MiB).

`cuda_model_range_ptr` (`rocm/ds4_rocm_runtime.cuh:4589`) resolves in order:
resident image -> exact-offset map -> containment scan -> **`_from_fd`**, which
allocates a fresh arena and copies from disk. The MoE launcher
(`rocm/ds4_rocm_moe_launch.cuh:706-708`) asks for the whole-layer range, the
first three lookups miss because TP mapped only half, so every layer copies in
the half this rank does not own. That is the +14.4 GiB, and it scales with
layers - it would have exhausted any headroom eventually.

**Patch 8 caused this.** It implements expert sharding as *weight masking*:
compute over all 256 experts, zero the weights of experts this rank does not
own (`ds4_tp_mask_weights_kernel`). Numerically exact - which is why it was
attractive - but it requires every expert to be **addressable**, which directly
contradicts TP mapping only half of them resident. The two are incompatible by
construction, and the masking kernel never even got to run.

## Why MXFP4 was the wrong response

MXFP4 gives 76.73 GiB/rank vs 80.76 (19.27 vs 15.24 GiB headroom). That is 4 GiB
against a leak proportional to layer count - it would have failed a few layers
later and looked like a different bug. The 156 GB transfer was done before this
was understood; the file is now on both nodes and costs nothing to keep, but it
does NOT fix this.

## The fix

Shard at **selection**, not after it: thread the owned expert range
(base, count) into the routed-MoE launcher so it only ever requests its own
half's offsets and never the whole-layer range. This
- removes the duplicate residency entirely (each rank touches only what it maps),
- deletes `ds4_tp_mask_weights_kernel` and its scratch buffer,
- and is what the Metal reference does - `ds4_metal.m:8327-8342` documents an
  expert *range*, not a mask.

Two upstream robustness bugs seen along the way, worth reporting separately:

1. `cuda_model_cache_limit_bytes` returns `UINT64_MAX` outside SSD-streaming
   mode (`ds4_rocm_runtime.cuh:5438`), so the model cache is uncapped and will
   allocate until the device is full.
2. On arena failure the NULL return is dereferenced rather than handled, turning
   a clean OOM into "an illegal memory access was encountered" plus a rocBLAS
   handle-destructor error.
