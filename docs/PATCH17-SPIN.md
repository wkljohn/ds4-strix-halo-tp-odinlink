# Patch 17: the service-thread back-off was costing ~25% of decode

## What was wrong

`ds4_tp_service_thread` (ds4_rocm.cu) backed off between flag checks with:

    for (int i = 0; i < (1 << 14); i++) __builtin_ia32_pause();
    if (!gpu_flag[0] && !gpu_flag[1]) sched_yield();

Two defects:

1. **16384 PAUSE is ~200-300 us of BLIND time per miss** on Zen (PAUSE ~40-65
   cycles). During that window an arrived gate sits unnoticed. With up to 86
   gates per token on both ranks, that is a large share of a ~100 ms token - and
   it is pure *detection* latency, not work.

2. **The yield guard was dead code.** It tested `gpu_flag == 0`, but the gpu
   flags are MONOTONIC sequence counters: after the very first gate they are
   never zero again. So `sched_yield()` was unreachable and the thread spun a
   core forever.

Fixed: back-off 1<<8 (~3-5 us), and yield on a miss STREAK (64 consecutive),
which is what the guard was trying to express. Also cached two per-gate
`getenv()` calls in ds4_tp.c (linear scans of environ on the gate path).

## Measured (RDMA, -n 100, --temp 0, same prompt, 3 runs)

    generation: 11.09, 10.84, 9.81 t/s   -> mean ~10.6, spread +-12%
    before patch 17: 8.57 t/s (ONE sample)

So roughly **+23%**, though the magnitude is uncertain: the post-patch spread is
+-12% and the pre-patch figure is a single sample. All three runs produced the
correct answer.

**~10.6 t/s now exceeds llama.cpp's no-draft 9.42 t/s** on this hardware,
and is ~42% of our own 25.1 t/s bandwidth ceiling (was ~30%).

## Do not quote prefill t/s at this prompt size

Prefill across the same three runs: 8.93, 11.38, 22.87 t/s - a 2.5x spread on
identical settings. The prompt is ~13 tokens, so prefill throughput is dominated
by fixed costs and is meaningless here. Any prefill claim needs a long prompt.


## Stable baseline (500-token runs)

The 100-token samples spread 9.81-11.09 (+-12%). At 500 tokens:

    run a: generation 10.46 t/s
    run b: generation 10.62 t/s

**~10.5 t/s is the trustworthy figure.** Use >=500-token runs for any future
comparison; 100-token runs are too noisy to distinguish changes below ~15%.

Prefill on the same two runs: 24.00 and 10.37 t/s - still a 2.3x spread even at
500 generated tokens, because prefill is set by the (short) prompt, not by -n.
Prefill needs a long PROMPT to be measurable at all.
