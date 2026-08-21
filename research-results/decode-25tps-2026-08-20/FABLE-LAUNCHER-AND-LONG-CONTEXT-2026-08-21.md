# Fable reviews: launcher precedence and long-context scaling

Date: 2026-08-21

## RDMA launcher precedence

An explicit command-line environment selection of OdinLink was overwritten by
plain assignments in `bench.env.local`, causing a correctly self-identified
RoCE v2 run. The result was not mislabeled, but the model load was wasted.

Fable recommended treating the local file as defaults and restoring all
explicitly exported `DS4_*` invocation values after sourcing it, using Bash
indirect expansion and `printf -v` rather than `eval`. It also identified two
adjacent fail-closed requirements:

- reject an explicitly empty prefill chunk, which would otherwise bypass the
  ConnectX-4 Lx 2,048-row registration safety default;
- print resolved provider/address/device/chunk values in validate-only mode and
  record the config identity and coordinator address in each manifest.

The first regression proof found a second mixed-profile hazard: the explicit
OdinLink profile initially retained `mlx5_0`/`mlx5_1` device names from the
RoCE-local config. The final implementation clears provider-specific device,
address, and GID values whenever an explicit profile differs from the config,
unless that value was itself explicitly supplied. Provider device defaults are
then recomputed and a missing address fails before model loading.

`tests/test_bench_env_precedence.sh` covers provider and address precedence,
config defaults, safe RoCE chunking, explicit chunking, invalid/empty values,
and the existing ambient expert-split rejection without SSH or model loading.

## Long-context scaling review

Measured Q4_K RoCE v2 data:

| Frontier | Prefill | Decode | Steady decode | First decode |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 257.87 t/s | 19.32 t/s | approximately 19.3 t/s | approximately 50 ms |
| 32,768 | 209.70 t/s | 16.29 t/s | 16.35 t/s | 103.249 ms |

Fable's quantitative interpretation:

- steady decode latency rises from about 51.76 to 61.16 ms/token, adding about
  9.4 ms for roughly 30,700 more visible tokens;
- 32K prefill takes sixteen sequential 2K chunks averaging about 9.77 s each,
  versus 7.94 s for the 2K baseline;
- the roughly 42 ms first-decode-only excess is likely first-touch, graph, or
  compressor-finalization work and should be measured separately from steady
  throughput.

The minimum next matrix is 2K/8K/16K/32K with three 32K repeats, per-chunk
prefill time, per-stage decode time, logged raw/compressed KV rows and bytes,
page-fault/SVM evidence, clocks, and transport byte counters. A 33,792 context
allocation with only a 2K prompt separates allocation-size effects from scan
length.

The leading no-quality-change architecture candidate is sequence-sharded
decode attention: each rank scans a disjoint history range and exchanges the
small online-softmax `(max, denominator, partial output)` state for an exact
log-sum-exp merge. This is justified only if the measurement matrix confirms
that the context-dependent attention scan owns the added latency. A local
split-KV flash-decoding kernel is secondary if achieved scan bandwidth is well
below the UMA streaming ceiling. Decode graph/KV pre-touch can separately hide
the first-token penalty.
