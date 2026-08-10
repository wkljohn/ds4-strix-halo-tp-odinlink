# DSpark ROCm TP Handover: Stable >17 t/s

## Objective

Reach **more than 17 generation tokens/s** with DSpark on the two-node Strix
Halo TP=2 setup, while preserving the existing good non-DSpark mode and adding
a selectable `dspark-good` mode only after the result is correct and repeatable.

Do not treat one outlier above 17 t/s as completion. Use at least three
identical warm runs in one loaded process and require the warm median to exceed
17 t/s with identical target output.

## Repository and machine layout

- Local repository: `/home/wkljohn/Desktop/cc/ds4-strix-halo-tp`
- Peer alias: `peer`
- Peer repository: `/home/wkljohn/Desktop/cc/ds4-strix-halo-tp`
- The filesystems are **not shared**. Copy changed sources or the exact binary
  to the peer and verify hashes before every two-node test.
- Start worker and coordinator immediately/concurrently. Never wait for the
  coordinator port before launching the worker.
- The user repository is `origin`; preserve the separate upstream-sync path.
- Worktree is intentionally dirty with the complete ongoing ROCm/TP/DSpark
  implementation. Do not reset or discard unrelated changes.

OdinLink environment:

```bash
DS4_TP_VERBS_LIB=/home/wkljohn/Desktop/cc/OdinLink-Five/build/verbs/libodl_tb5_verbs.so.0.1.0
LD_LIBRARY_PATH=/home/wkljohn/Desktop/cc/OdinLink-Five/build/lib:/home/wkljohn/Desktop/cc/OdinLink-Five/build/verbs
```

Do not use `LD_PRELOAD`. Keep OdinLink-specific behavior gated so generic
verbs/Mellanox remains unaffected.

## Standard DSpark test configuration

The requested standard placement is 46/54 because the coordinator also runs
the drafter:

```bash
DS4_TP_EXPERT_SPLIT=118       # rank 0: 118/256, rank 1: 138/256
DS4_ROCM_TP_SKIP_UNOWNED=1
DS4_DSPARK_RESIDENT_Q8=1     # explicit opt-in; carries a ~10.15 GiB warning
DS4_DSPARK_SCHEDULER=0        # main confidence/scheduling knob stays disabled
DS4_DSPARK_SUPPORT_TOPK=4
DS4_DSPARK_MAX_DRAFT_TOKENS=5
DS4_DSPARK_STATS=1
```

Target-only/non-DSpark remains 50/50. Do not globally change it to 46/54.
Resident Q8 must remain opt-in because of its memory footprint.

Reference prompt and generation:

```text
context: 128 for the short benchmark
temperature: 0
seed: 42
thinking: disabled
generation: 60 tokens
prompt: Write a Python function that returns the factorial of a non-negative integer.
```

Use `/reset` between repetitions. `/ctx` recreates KV but retains the chat
transcript and therefore is not a valid identical-run reset. `/reset` was added
to `ds4_cli.c` and `ds4_help.c` specifically for this benchmark.

## Current measured performance

The established non-DSpark decode result is **13.83 t/s**.

| Configuration | Decode | Important timing/evidence |
|---|---:|---|
| Width 5, compact Q8 small-token tile | 12.36 t/s | verifier 3271.6 ms, proposal 382.3 ms, replay 272.9 ms, target 847.6 ms |
| Same, Q8 K-block tile 32 | 12.58 t/s | small positive; verifier 3178.8 ms |
| DP4A + four-prefix commit, repeated warm run | 14.92 t/s | verifier 2845.6 ms, proposal 311.2 ms, replay 0, target 849.5 ms |
| Same repeated run | **15.21 t/s** | verifier 2786.2 ms, proposal 316.3 ms, replay 0, target 825.3 ms |
| Exact-five DP4A specialization, three identical runs | 14.81 / **15.12** / 14.88 t/s | warm median still below 15 |
| Experimental 16-wave DP4A workgroup | 11.86 / 12.04 / 15.03 t/s | unstable and rejected |

The best repeatable state is approximately **15 t/s**, not 17 t/s. To reach
17 t/s, a 60-token run must finish in under 3.529 seconds. The stable ~15 t/s
breakdown is approximately:

```text
target verifier     2.80 s   dominant
target anchors      0.84 s
DSpark proposals    0.31 s
partial replay      0.00 s
snapshots/readback  ~0.01 s
```

The remaining target is roughly **0.45 seconds**, and most of it must come from
the verifier.

## Implemented improvements

### 1. Asymmetric TP expert placement

`DS4_TP_EXPERT_SPLIT=118` is encoded in the TP hello feature word, must match
on both ranks, and is used by model-span warming, expert ownership, and runtime
mapping. Rank 0 warms 75.09 GiB; rank 1 warms 86.43 GiB. TP hello tests cover
matching and mismatched splits.

### 2. Skip known-unowned Q4_K experts

`DS4_ROCM_TP_SKIP_UNOWNED=1` maps unowned routed experts to a sentinel and
skips their local small-row gate/down work. It was a modest but real win.

### 3. Small-row Q8 token tiles

`DS4_ROCM_Q8_SMALL_BATCH_TILE=1` selects token tiles 2/4/8 instead of always
reserving a 32-row LDS tile. This was the largest earlier improvement:
approximately 8.96 to 12.36 t/s on the reference run.

### 4. Native packed INT8 DP4A path

`DS4_ROCM_Q8_SMALL_BATCH_DP4A=1` dynamically quantizes 2–8 activation rows to
Q8 and applies a token-reuse DP4A kernel to Q8_0 weights. The production kernel
is in:

```text
rocm/ds4_rocm_q8.cuh
rocm/ds4_rocm_matmul.cuh
```

The standalone harness is:

```text
scripts/q8_small_batch_dp4a_bench.cu
```

It measured 1.98–3.54x over the old Q8-weight/F32-activation microkernel on
actual DSpark shapes, including activation quantization, with normalized RMS
error around 0.38%. Generated target text and the original acceptance histogram
were unchanged in the first integrated A/B.

ISA was extracted and confirmed to contain native `v_dot4_i32_iu8`; this is not
a scalar fallback. The source uses the existing `__dp4a` abstraction.

### 5. Four-prefix verifier frontier snapshots

The verifier now captures compressor/indexer frontiers for accepted prefixes
1–4. Arbitrary partial accepts commit the matching frontier rather than replay
accepted target tokens. Measured replay time fell from about 0.28 seconds to
zero.

This adds small state snapshots, not model weights. Estimate is below 100 MiB,
but measure actual VRAM delta before promotion.

Important correctness concern: after enabling all-prefix commit, the same
prompt changed from 11 verifier cycles with acceptance histogram
`1:1,3:1,4:1,5:8` to 12 cycles with `1:2,3:1,4:2,5:7`. Target output remained
identical because verification is exact, but the support-ring/frontier state may
not exactly match the old replay reference. Validate draft tokens and support
cache windows after each partial commit before making this default.

### 6. `/reset`

Interactive `/reset` creates a genuinely fresh transcript/session without
reloading the engine. Use it for identical repeated measurements.

## Current worktree warning

The current source at handover uses **16 waves / 512 threads** in
`cuda_launch_q8_small_batch_dp4a()`. That setting was the final experiment and
was unstable end-to-end despite a small microbenchmark win.

Before continuing, revert only that launch shape to the proven setting:

```cpp
rows_per_block = 8u;
kernel launches = 256 threads;
```

Keep the exact 3-row and 5-row specializations and native
`__float2int_rn()` quantization while testing. Do not discard the broader
worktree.

`DS4_DSPARK_CHAIN_CYCLES=1` also remains experimental. It was made logically
correct with a verifier-capture window conversion, but measured only 6.56 t/s
because back-to-back verifier batches lost the useful target-weight warm-up.
Do not enable it in `dspark-good`.

## Rejected experiments

- Support top-k 2: 7.36 t/s, acceptance fell to 81%, proposal time worsened.
- Q8 K-block tile 8: 7.78 t/s, verifier rose to 6.19 seconds.
- Chained speculative cycles: 6.56 t/s after correctness fixes.
- 16-wave/512-thread DP4A production launch: severe verifier-time variance.
- Confidence scheduling as the main control: explicitly disallowed; keep
  `DS4_DSPARK_SCHEDULER=0`.

## Immediate next steps toward >17 t/s

1. Revert DP4A workgroups to 8 waves/256 threads and rebuild/sync both nodes.
2. Inspect and set both nodes to a reproducible performance profile before
   benchmarking. At handover, node 1 reported:

   ```text
   power profile: balanced
   CPU governor: powersave
   GPU DPM level: auto
   ```

   This likely contributes to 2.8 vs 3.8 second verifier swings. Verify both
   nodes independently; their filesystems and state are not shared. Record the
   profile in every result log. Do not silently require a destructive or
   machine-wide setting in normal inference.
3. Run three identical `/reset` trials with the 256-thread DP4A build and use
   the warm median, not the best sample.
4. Validate four-prefix support state against replay mode. Log draft tokens,
   cache token start/length, and compressor counters after accepted prefixes
   1, 3, and 4. Fix any mismatch while retaining zero replay.
5. Profile the target verifier, which still consumes ~70% of runtime. Focus on
   the per-layer Q8 dense projections and TP MoE, not proposal confidence.
6. Consider separating DP4A dispatch by tensor shape. The microbenchmark found
   different optimal workgroup sizes: 16 waves helped output A/B and FC, while
   Q-B was flat/slightly better at 8 waves. A shape-specific launch may recover
   a few percent without instability.
7. Investigate a fused activation-quantize + DP4A path or reuse one quantized
   activation across compatible projections. The current generic call
   requantizes its input for every Q8 matmul.
8. Measure whether target anchor slowdowns correlate with node power/fabric
   clocks, memory pressure, or drafter-induced cache eviction. Do not remove
   target anchors blindly: they appeared to warm target weights and made the
   following verifier faster.
9. Only after the warm median exceeds 17 t/s, create `dspark-good` with:

   ```text
   split 118/138
   top-k 4
   draft width 5
   scheduler/confidence main knob off
   validated DP4A path
   validated all-prefix commit
   resident Q8 explicitly selected with its ~10.15 GiB warning
   ```

10. Verify non-DSpark mode remains near 13.83 t/s and that DP4A/all-prefix
    switches do not affect it when DSpark is disabled.

## Build and sync

Use the ROCm target or the equivalent incremental link variables already used
in this worktree:

```bash
make -j4 strix-halo
scp -q ds4 peer:/home/wkljohn/Desktop/cc/ds4-strix-halo-tp/ds4
sha256sum ds4
ssh peer 'sha256sum /home/wkljohn/Desktop/cc/ds4-strix-halo-tp/ds4'
```

Start the peer worker detached with `setsid -f`, then immediately start the
local coordinator. If a failed run leaves processes alive, terminate only the
exact coordinator/worker command, not broad system processes.

## Evidence location

All relevant logs are persistent and intentionally outside `/tmp`:

```text
research-results/dspark-resident-2026-08-07/clean-width-sweep/
```

Most useful files:

```text
coordinator-width5-small-q8-tile.log
coordinator-width5-q8-bt32.log
coordinator-width5-q8-dp4a.log
coordinator-reset3x-dp4a-prefixes.log
coordinator-reset3x-dp4a-exact5.log
coordinator-reset3x-dp4a-rpb16.log
```

## Verification still required

- Rebuild and rerun the broader ROCm test binary after the latest changes.
- Add a production-path correctness test for Q8 DP4A at 2, 3, 4, 5, and 8
  rows, including odd output tails and all important DSpark shapes.
- Confirm exact generated text over several prompts, not only factorial.
- Measure VRAM delta of four-prefix snapshots.
- Verify no-DSpark and generic verbs/Mellanox behavior with all new switches
  absent.
- Create/document `dspark-good`; it does not exist yet.
- Commit and push only after the >17 t/s warm-median acceptance gate passes.

## Last known tests and process state

- `tests/test_tp_hello` passed, including asymmetric split and OdinLink versus
  generic/Mellanox message policy checks.
- Local and peer `ds4` binaries matched at handover.
- No coordinator or worker process was active when this note was written.
- No final commit/push has been made for this DSpark optimization series.
