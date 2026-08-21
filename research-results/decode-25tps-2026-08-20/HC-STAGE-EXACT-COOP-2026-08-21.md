# Exact cooperative HC decode stage

Date: 2026-08-21

## Question

Can the ordinary one-token HC pre-chain remove launch boundaries without a
weight cache or a numerical change?

The established chain is:

1. 256-thread F32 RMSNorm over the 16,384-value HC residual;
2. 24-row ordered-chunk F16 projection;
3. fused HC split, weighted sum, and weighted RMSNorm.

## Model-free oracle

`scripts/hc_stage_exact_coop_bench.cu` reproduces the shipped arithmetic and
compares it with a 24-block cooperative kernel. The accepted implementation
retains the existing normalized F32 scratch. `split[0]` temporarily stores the
RMS scale, then the established split function overwrites it before it is
observable.

On gfx1151, a cooperative-grid residency probe reported 160 resident blocks
for the 24-block grid. The exact oracle reported bitwise-identical flat, mix,
split, output, and normalized output buffers.

| Path | Mean per HC chain | Change |
| --- | ---: | ---: |
| Existing three launches | 48.395 us | baseline |
| Cooperative, scratch-preserving | 36.697 us | -24.2% |
| Cooperative, recompute normalized input | 50.745 us | +4.9% |

There are 86 applicable chains per generated token, projecting approximately
1.01 ms/token saved before whole-model effects.

## Production candidate

The production path is opt-in with `DS4_ROCM_HC_STAGE_EXACT_COOP=1`. It is
restricted to gfx1151, the exact 4096x4 HC layout, F16 16384x24 projection
weights, ordinary non-DSpark TP=2 decode, and hardware with enough cooperative
grid residency. A TP hello feature bit makes asymmetric rank selection fail
closed. It allocates no persistent or temporary buffer beyond the existing
scratch tensors.

Gold-standard Q4_K RoCE v2 candidate:

| Tag | Prefill | Decode | Fingerprint | Semantic suite |
| --- | ---: | ---: | --- | --- |
| `q4-hc-exact-coop-r1` | 258.06 t/s | 19.54 t/s | `5f8a983422299d76` | pass |

Workload: 2,048-token fixed frontier plus 300 generated tokens, balanced
128/128 experts, mandatory RoCE v2, cache-free. The exact fingerprint matches
the established Q4_K production baseline. The arithmetic and 1,532-token
retrieval semantic cases both passed.

The prior three-run Q4_K RoCE v2 median was 257.87 prefill / 19.32 decode t/s.
One candidate run is sufficient to admit the implementation as a tested stage,
but repeat medians and Q2_K/OdinLink provider checks remain required before it
can become a default.
