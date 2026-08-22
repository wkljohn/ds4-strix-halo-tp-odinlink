# Rejected exact TP-split Q8 verifier output head

## Decision

This experiment is **rejected for the production candidate set**.  Splitting
the 129,280-row Q8 verifier vocabulary projection evenly across the two TP
ranks is arithmetically exact and nearly halves the isolated head time, but it
does not produce a repeatable improvement over the accepted 14.08 tok/s long
DSpark median.  The extra protocol and selected-row exchange are therefore not
worth carrying toward the 19.57 tok/s ordinary-decode milestone.

No persistent allocation or expanded-weight cache was added.

## Design tested

Each rank computed 64,640 output rows using the existing exact Q8 decode-row
kernel.  The worker sent up to four local `(token_id, value)` maxima to the
leader.  The leader merged them with the same lowest-token-ID tie rule as the
full argmax.  After the verifier commit, both ranks exchanged only the chosen
252.5 KiB half-row through the existing registered RDMA slab and reconstructed
the complete logits row before state commit.

The experiment used a new, exact-match TP feature bit and protocol version 10.
The control frame was fixed-size and drained before every commit, including a
failed local verifier.  A protocol-only switch retained duplicated full-head
computation while exercising the maxima frame.

## Isolated gate

`make test-rocm-q8-output-tp-split-exact` compared each half against the full
Q8 output, merged maxima, tie behavior, NaN behavior, and the selected row.

| Width | Full head + top-1 | Lower half | Upper half | Ideal parallel speedup | Exact |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 2 | 12.29 ms | 6.17 ms | 6.15 ms | 1.99x | yes |
| 3 | 12.99 ms | 6.53 ms | 6.51 ms | 1.99x | yes |
| 4 | 13.28 ms | 6.66 ms | 6.65 ms | 1.99x | yes |
| 5 | 13.79 ms | 6.91 ms | 6.89 ms | 2.00x | yes |

An unfused 2,048+60 model diagnostic directly timed 15 verifier heads.  The
full-head control used 92.111 ms; the split used 49.485 ms, a 46.3% reduction
or 2.842 ms per verifier.  Both produced fingerprint
`134e4c2205bd82a8` and the same acceptance histogram.

## End-to-end evidence

All rows below used mandatory RC/RoCE v2, the Antirez target and support model,
the 2,048-token frontier, exact arithmetic switches, and no weight cache.

| Run | Generated | Prefill | Decode | Fingerprint | Interpretation |
| --- | ---: | ---: | ---: | --- | --- |
| Protocol only, TCP target logits | 60 | 199.87 | 16.46 | `134e4c2205bd82a8` | Full head plus maxima frame |
| Protocol only, all target logits over RDMA | 60 | 200.04 | 16.77 | `134e4c2205bd82a8` | Short-run transport diagnostic |
| Split head, all target logits over RDMA | 60 | 199.01 | 16.82 | `134e4c2205bd82a8` | Exact short improvement |
| Protocol only, all target logits over RDMA | 300 | 189.21 | 13.88 | `2aa153138c195efc` | Long control exposed a transport coupling |
| Split head, all target logits over RDMA | 300 | 190.31 | 13.89 | `2aa153138c195efc` | Split was not the regression source |
| Split head, selective verifier-row RDMA r1 | 300 | 191.08 | 14.02 | `2aa153138c195efc` | Global RDMA-logit coupling removed |
| Split head, selective verifier-row RDMA r2 | 300 | 190.72 | 13.99 | `2aa153138c195efc` | Predefined acceptance became impossible |

The previous accepted exact temporal-F16 candidate measured 14.09, 14.00, and
14.08 tok/s, for a 14.08 median.  Before running the final pair, the acceptance
gate was fixed at a split median of at least 14.08 tok/s and at least the
matched long control.  After 14.02 and 13.99, no third result could make the
three-run median reach 14.08, so testing stopped early rather than extending
the sample until a favorable result appeared.

The first selective-RDMA run reduced the main 89-verifier total from the prior
14,481.764 ms median to 14,336.998 ms, or 1.63 ms per verifier.  That is only a
145 ms ceiling across roughly 21 seconds of generation (about 0.10 tok/s), the
same scale as the established run-to-run spread.  Variable upload and proposal
setup time consumed the saving.

Candidate mode passed the arithmetic and 1,532-token retrieval semantic cases.
The 300-token trajectory retained 153 cycles, 232 accepted drafts, the exact
acceptance histogram, and 1.961 final tokens per cycle.  There were no verifier,
transport, or fallback errors.

## Lessons

- The output head is not large enough in the long verifier wall time for this
  protocol to matter, even when its local compute is nearly halved.
- Do not make a verifier-only feature implicitly select RDMA for every ordinary
  target-logit row.  Short and long workloads chose different winners.
- The selected-row exchange itself is not the bottleneck: measured verifier
  readback was only about 9.5 ms total across 89 heads.
- Preserve isolated wins as evidence, but require the fixed long-context gate
  before accepting permanent distributed-protocol complexity.

## Artifacts

CSV files, manifests, and both-rank logs are in `runs/` under these tags:

```text
antirez-q4-dspark-outputtp-proto60-r1
antirez-q4-dspark-outputtp-proto-rdma60-r1
antirez-q4-dspark-outputtp-split60-r1
antirez-q4-dspark-outputtp-proto-rdma-unfused60-r1
antirez-q4-dspark-outputtp-split-unfused60-r1
antirez-q4-dspark-outputtp-split300-candidate-r1
antirez-q4-dspark-outputtp-proto-rdma300-control-r1
antirez-q4-dspark-outputtp-selective-rdma60-r1
antirez-q4-dspark-outputtp-selective-rdma300-r1
antirez-q4-dspark-outputtp-selective-rdma300-r2
```

Fable reviewed the first long regression and recommended the matched
protocol-only control, the 14.08 tok/s stop threshold, and rejection when the
effect could not rise above the existing noise floor.  A Grok review was not
available in this session because no Grok tool or MCP endpoint was exposed; no
review was fabricated.
