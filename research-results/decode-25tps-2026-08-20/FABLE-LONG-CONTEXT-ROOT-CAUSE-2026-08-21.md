# Fable review: long-context prefill and decode degradation

Date: 2026-08-21

Measured matched Q4_K RoCE v2 evidence supplied to the reviewer:

- 2K exact candidate median: 258.06 prefill t/s and 19.55 decode t/s.
- 32K run: 209.70 prefill t/s and 16.29 decode t/s; steady decode 16.35 t/s; first decode token 103.249 ms.
- The longer run adds about 9.4 ms/token to steady decode for roughly 30.7K extra visible tokens.
- Long prefill is processed as sequential 2K chunks.

## Unedited Fable response

context: caller prompt only

Verdict: **proceed-with-changes** — the measurement-first plan in FABLE-LAUNCHER-AND-LONG-CONTEXT-2026-08-21.md is right, but its leading architectural candidate (sequence-sharded *attention*) is aimed at the wrong sub-term. The code shows the indexed attention core is top-k-capped and therefore roughly constant in context; the context-scaling terms live in the **indexer score scan and the top-k selection**, and at exactly this workload the top-k crosses a hard implementation cliff.

The decisive code fact first: `rocm/ds4_rocm_indexer.cuh:909-936` — the fast single-block top-k paths stop at `n_comp <= 8192`. With ratio 4, a 32,768-token prefill gives `n_comp = 8192` at the frontier, so **the very first generated tokens fall off the fast path** (`n_comp = (32768+t)/4 > 8192` for t≥4) into the chunked bitonic tree at `ds4_rocm_indexer.cuh:1020-1081`: per layer per token, a `<4096>` chunk-sort kernel across 3 chunks, tree merges, a final 1024-thread bitonic merge, plus `cuda_tmp_alloc` scratch. Your 32K steady-decode measurement sits entirely on this slow path; a 30K measurement would not. Any interpretation of the +9.4 ms as "inherent context scan" is unsafe until this cliff is separated out.

**1. Prefill degradation, ranked**

1. **Indexer score work, O(chunk × visible n_comp × 64 heads × 128 dims)** — per chunk it is a dense score matrix (`indexer_scores_wmma128_kernel`, `ds4_rocm_indexer.cuh:128-252`, launched at `:792-800`), linear in chunk index; summed over 16 chunks it is the quadratic term (the DSA lightning-indexer cost, divided by ratio). Predicts a roughly linear per-chunk ramp: last chunk ≈ 7.94 s + 2×(9.77−7.94) ≈ 11.4 s.
2. **Batched top-k over growing n_comp** (`ds4_gpu_indexer_topk_tensor`), with kernel-family changes at n_comp 1024/2048/4096/8192 (`:868-936`) — predicts *steps* at chunk boundaries 2/4/8/16, superimposed on the ramp. Also the score-matrix traffic itself: final chunk writes+re-reads 2048×8192×4B = 64 MiB per layer per top-k pass.
3. **Not attention**: prefill attention is the indexed heads8 online kernel with ≤256 raw + ≤`top_k` comp rows (cap 1024, `ds4_rocm_attention.cuh:7`, `:1038-1199`) — constant per token. The uncapped static kernel bails at 768 rows (`:1238-1239`), so it isn't the long-chunk path.
4. Per-chunk fixed costs (TP gates, RDMA exchange, compressor batch) are constant × 16 — they explain sequentialization overhead but not the 7.94→9.77 s growth.

**2. Steady decode degradation (+9.4 ms/token ≈ +219 µs/layer over ~43 layers), ranked**

1. **Top-k slow path** (see above). 5+ extra kernel launches/layer, bitonic sorts with dozens of `__syncthreads` sweeps, likely 100–250 µs/layer → alone can account for most or all of 9.4 ms. Implementation artifact.
2. **Indexer score scan**: `indexer_score_one_direct_kernel` (`:85-126`) — one 128-thread block per comp row, F32 `index_comp`, each block re-reading all 64 q heads (32 KB) from cache. Inherent bytes: extra ~7.7K rows × 512 B × 43 layers ≈ 169 MB/token → **≈0.9 ms at the 195.6 GiB/s UMA ceiling**. Anything above ~1 ms attributed to this kernel is achieved-bandwidth shortfall (launch geometry, q re-reads), i.e., fixable.
3. **Duplicated across ranks**: the decode indexer kernel hard-codes all 64 heads (`for (h0 < 64)`, `:114`) — both TP ranks appear to do the full linear scan redundantly. Artifact/architecture-boundary cost.
4. Attention core: constant (top-k capped). Score-buffer writes (33 KB/layer): negligible. Rank gate-wait skew (138 µs/layer at 2K per SURVEY §6) may shift with context but is not obviously context-scaled — measure, don't assume.

**3. First decode token (103.2 ms vs ~61 ms steady)**

- First entry into the tree top-k grows the global scratch arena: `cuda_tmp_alloc` (`ds4_rocm_runtime.cuh:581-600`) does `cudaFree`+`cudaMalloc` on growth — `hipFree` is device-synchronizing; if any later caller (e.g., "attention output a warmup", `:5688`) needs more bytes, the arena thrashes once.
- Deferred compressor/indexer finalization of the prefill tail (temporal ratio-4/128 batching) landing on token 1.
- First-touch/migration of managed memory: KV allocations are `cudaMallocManaged` with an 8–40 GiB reserve heuristic (`ds4_rocm_runtime.cuh:6468`, `:6504-6509`) — first decode touch of pages written by prefill kernels in a different access pattern can fault/migrate.
- First code-object load of the tree-merge kernels never used at 2K.
These are one-time; FABLE's instinct to time it separately is correct.

**4. Inherent vs artifact**

- Inherent (architecture, survives any implementation): decode-side linear indexer scan ≈ 0.9–1.5 ms/token at 32K; prefill-side quadratic indexer score FLOPs/bytes (÷ratio); compressed-KV growth in memory.
- Artifact (fixable without architecture change): top-k >8192 cliff and its launch storm; scores-kernel achieved bandwidth; per-rank duplication of the indexer; arena regrow + finalization on token 1; per-branch `cudaDeviceGetAttribute/cudaFuncSetAttribute` calls executed **every** top-k invocation in the 4096–8192 paths (`:884-896` etc.).
- The measured 9.4 ms is ~6–10× the inherent decode floor. That ratio is the whole story.

**5. Minimal measurement matrix**

- Contexts: 2K, 8K, 16K, **30K and 33K** (straddling the n_comp=8192 branch), 32K ×3. The 30K-vs-33K delta isolates the top-k cliff with everything else nearly equal.
- Keep FABLE's 33,792-allocation/2K-prompt control (separates allocation/managed-reserve effects from scan length).
- Per-layer decode stage timers splitting **indexer-score / top-k / indexed-attention / compressor / MoE / gate-wait** — the existing stage harness (SURVEY §6 table) lumps or omits indexer-score and top-k; that split is the single most important new number.
- Log per call: `n_comp`, which top-k branch fired (the `cuda_ok` tag strings at `:872-1085` are ready-made labels), `cuda_tmp_alloc` grow events with bytes (currently only failures print, `:592`).
- rocprof achieved read bandwidth for `indexer_score_one_direct_kernel` vs the 195.6 GiB/s ceiling; kernel launch count per token at 2K vs 32K.
- Per-chunk prefill wall time (16 values) fitted to linear ramp + steps at chunks 2/4/8/16.
- First-token decomposition: time from last prefill gate → first indexer launch (finalization), first top-k call duration vs steady, arena grow events.
- Both ranks: per-layer gate-wait skew histogram at 2K vs 32K; RoCE port byte/packet counters per phase (should be context-flat for decode — confirm).

**6. Specific suspicions**

- Rank imbalance: secondary; but confirm indexer duplication (both ranks, 64 heads) — it doubles the inherent term.
- RoCE/OdinLink: decode transport should be context-invariant (~6 µs paired min, 86 gates); the 2K prefill chunk is a ConnectX-4 Lx registration safety bound (`run-tp-ds4-bench.sh:308`), not a perf lever — don't blame the wire without counter deltas.
- SVM: managed KV + reserve heuristic is real (`:6468`, `:6504`); check `xnack` page-fault counters during token 1 specifically.
- KV layout: comp_kv is F32, 2 KB/row; fast paths refuse F16 (`ds4_rocm_attention_launch.cuh:559,587`) — halving index_comp/comp_kv width would halve the inherent scan, but only after the artifact terms are gone.
- Logging: per-branch attribute queries in hot top-k paths, and the `topk_mask_kernel` (`ds4_rocm_indexer.cuh:745-759`, O(n_comp×top_k) per row) if the masked path is ever taken at long context — verify it is not.

**7. Stop/go for sequence-sharded attention**

Go only if all three hold after the matrix: (a) 30K-vs-33K shows the top-k cliff explains <3 ms of the 9.4 (i.e., the cliff fix alone can't recover most of it); (b) the indexer-score scan measures ≥4 ms/token at 32K *after* confirming ≥60% of UMA ceiling (i.e., it's inherent bytes, not kernel inefficiency); (c) the sharded exchange budget closes — note the cheap variant shards the **indexer** (exchange 512 indices+scores ≈ 4 KB/layer, trivial over the 6 µs gates), not attention partial outputs (64 heads × 512-dim partials ≈ 131 KB/layer/token, ~2 ms/token at RoCE bandwidth, which could eat the entire saving). If (a) fails — likely, on this evidence — fix the top-k path and re-measure before any architecture work.
