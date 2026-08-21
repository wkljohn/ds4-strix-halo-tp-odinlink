# Exact Q4_K verifier down first-owner reuse

Date: 2026-08-22
Branch: `research/dspark-antirez-q4-32tps-20260821`

## Result

The accepted candidate reuses each selected Q4_K down-projection row across
matching verifier tokens while preserving the shipped direct-sum6 arithmetic.
It adds no persistent allocation or expanded-weight cache.  The fixed
2,048+300 Antirez Q4 workload retained FNV64 `7174e214e05fd83e` and the exact
acceptance histogram
`21,5,11,5,3,5,1,3,2,2,2,1,0,0,0,0,4` in every production run.

| Run | Provider / mode | Prefill | Decode | FNV64 |
|---|---|---:|---:|---|
| r1 | RoCE v2 + DSpark | 189.60 t/s | 15.04 t/s | `7174e214e05fd83e` |
| r2 | RoCE v2 + DSpark | 190.62 t/s | 15.04 t/s | `7174e214e05fd83e` |
| r3 | RoCE v2 + DSpark | 188.68 t/s | 15.07 t/s | `7174e214e05fd83e` |
| Cross-provider | OdinLink + DSpark | 181.67 t/s | 15.19 t/s | `7174e214e05fd83e` |
| Ordinary control | RoCE v2, no DSpark | 257.21 t/s | 19.68 t/s | `b7694f9d11a3760e` |

The production RoCE v2 median is **189.60 prefill / 15.04 decode t/s**.  Decode
is 1.3% above the preceding exact batched-argmax median of 14.85 t/s and 15.4%
above the 13.03 t/s first-owner baseline.  The ordinary path did not advertise
or select this DSpark-only feature and remained above its 19.57 t/s reference.
OdinLink reported 37,908 streaming calls, 3.49 GB, and zero fallback calls on
both ranks.

This is a valid incremental verifier win, not the milestone: DSpark remains
4.53 t/s below ordinary TP=2 decode.

## Arithmetic design

The shipped exact direct-sum6 path does not merely add one completed dot to the
previous slot total.  Its emitted gfx1151 instruction order folds the prior
slot total between the two quarter-wave halves:

```text
(left_half + prior_total) + right_half
```

The first prototype computed a completed partial first and then folded it.  It
was fast but mismatched 3,840 of 8,192 W2 outputs (maximum absolute error
3.81e-6), so it was rejected before integration.

The corrected partial kernel stores both reduction halves independently for
each route and output row.  The fold kernel walks slots in their original
order and recreates `(left + total) + right` with compiler barriers.  Gate,
up, mid, and the unused aligned tail of the existing down workspace provide
transient scratch; all are dead at this point.  No model-sized or persistent
buffer is introduced.

TP maps peer-owned routes to zero weight.  The first live version still
scanned and zero-filled those routes even though the exact fold skips them.  It
regressed from a paired 14.95 t/s control to 14.46 t/s.  The accepted version
passes the existing `DS4_ROCM_TP_SKIP_UNOWNED` policy into the partial kernel
and returns immediately for peer-only zero-weight owners.  That three-line
guard removed the deferred GPU backlog: verifier time fell to a three-run
median of 13,417.7 ms over all 83 full verifiers, rather than moving work into
the next synchronous upload bucket.

## Isolated correctness and speed gate

The production-linked oracle covers widths 2--5, mixed, disjoint, and
all-shared routing, a real local expert zero, TP-style zero-weight remaps, and
skip-unowned both enabled and disabled.  Every final output is bit-exact.

| Case | Full routed-MoE speedup | Final mismatches |
|---|---:|---:|
| W2 mixed | 1.50x | 0 / 8,192 |
| W3 mixed | 1.79x | 0 / 12,288 |
| W4 mixed | 2.03x | 0 / 16,384 |
| W5 mixed | 2.41x | 0 / 20,480 |
| W2 disjoint | 1.34x | 0 / 8,192 |
| W5 all-shared | 2.61x | 0 / 20,480 |

A 64-sample synchronized isolated W5 profile measured the down stage at
0.497 ms before and 0.352 ms after, a 1.412x speedup.  The complete isolated
MoE improved 1.124x.

## Live GPU-event attribution

Host `verify_layer` timing ends after asynchronous command submission, so the
next synchronous upload can inherit unfinished work.  Promotion therefore
used compiler-gated GPU events in `PROFILE=1` binaries on both ranks.  These
runs are diagnostic and are not included in the production median.

| Live stage | Candidate off | Candidate on | Change |
|---|---:|---:|---:|
| Rank 0 W2 down | 0.164 ms | 0.138 ms | -15.9% |
| Rank 1 W2 down | 0.173 ms | 0.149 ms | -13.9% |
| Rank 0 W5 down | 0.305 ms | 0.259 ms | -15.1% |
| Rank 1 W5 down | 0.335 ms | 0.294 ms | -12.2% |
| Rank 0 routed MoE | 1.379 ms/layer | 1.338 ms/layer | -3.0% |
| Rank 1 routed MoE | 1.502 ms/layer | 1.453 ms/layer | -3.3% |

The stage-event ring recorded all 3,569 verifier layers with zero drops on
both ranks.  The production binary compiles this instrumentation out.

## Reproduce

Use the standard command from `Q4-FIRST-OWNER-20260821.md`, retain the exact Q8
attention, two-head attention, and batched-argmax switches, and add this switch
symmetrically:

```sh
export DS4_ROCM_Q4K_VERIFY_DOWN_FIRST_OWNER=1
```

Run three distinct `ds4-bench-tp` tags with candidate mode and expected FNV64
`7174e214e05fd83e`.  The path is shape-gated to the exact DSpark Q4_K verifier
geometry and negotiated through TP feature bit 25; mismatched ranks fail
closed.

Raw logs, CSV files, and manifests are under `runs/` with tags beginning
`antirez-q4-dspark-down-first-owner-shard-guard-`.  The paired rejected run,
paired control, and compiler-gated event A/B are retained beside them.

## Next bottleneck

The accepted kernel removes only about 3% of routed-MoE GPU time.  At the fixed
2.098 final tokens per cycle, DSpark is still slower than ordinary inference.
The next work must target a materially larger verifier family or improve
accepted tokens per cycle without changing target-model quality; another
small down-kernel tweak cannot close the 4.53 t/s gap.
