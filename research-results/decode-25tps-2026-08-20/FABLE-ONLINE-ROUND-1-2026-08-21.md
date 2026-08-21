## Round 1 — adversarial ranking

**Verdict: proceed-with-changes** — the packet's "strongest local plan" (profile chains → HC fusion first) is ordered wrong. The ledger you already own proves launch-chain fusion cannot close 4.14 ms even if every fusion works; the only pool of the required size is the FFN gate-wait/skew pool, so the completion-word ordering probe is the highest-information experiment, not HC fusion.

### The arithmetic that constrains everything

From your own artifacts (verified locally):

- `FABLE-REVIEW-6-REMAINING-GAP.md`: the non-bandwidth gap decomposes as ~1.5 ms wire floor (immovable), **~2.4 ms FFN jitter idle** (43 gates × ~57 µs), ~0.5–1.0 ms attention static skew, ~1.7 ms boundary projection (side-stream overlap already rejected).
- PR 67's own numbers: 58.73→59.53 t/s = 0.23 ms/token for 160 launches removed ≈ **1.4 µs/launch**. Your HIP-graph exclusion independently measured <2 µs/launch. So every launch-count mechanism on this list is priced at ~1.5–2 µs × launches removed. With ~43 routed layers, a one-launch-per-layer fusion is worth ~0.06–0.1 ms; the whole per-layer small-kernel population is a pool of at most a few tenths of a ms unless a fusion also removes a gate or a memory round-trip that matters.

That single number is the adversarial filter for four of the six candidates.

### Ranking

**Exact-arithmetic candidates**

1. **Gate-free FFN row sharding (Mechanism I fold, enabled by a device-visible completion-word ordering probe).** Boundary: routed-expert mid/down exchange at the existing FFN gate, replacing the second `hipLaunchHostFunc` that GATE-8 denied. Max ≈ 2.4 ms (full jitter pool); likely 1.5–2.2 ms (your own Fable ceiling 20.1–20.4 t/s ⇒ 2.0–2.7 ms). Persistent memory ≈ 0 (ownership masks + completion words). TP interaction: it *is* the TP mechanism; the risk is RDMA write-ordering, not arithmetic — the `7787694` oracle is already bitwise on the fold. Smallest oracle: two-node probe where the NIC writes payload-then-word and a spinning kernel polls the word and checksums the payload, under concurrent gate traffic, **run on both RoCE v2 and OdinLink separately** — a RoCE-only pass proves nothing about OdinLink's ordering semantics.
2. **Entrpi HC two-kernel fusion (`135ab507`, claim −2.0 ms).** Boundary: RMS → F16 HC projection → fused tail, ×2/layer (confirmed: `hc_split_weighted_sum_norm_fused_kernel` exists as the tail; RMS and projection are separate launches in `ds4.c`). Adversarial pricing: merging 3→2 launches twice per layer removes ~86 launches ≈ 0.15 ms of launch cost; the 7168-float intermediate round-trip is ~28 KB/layer ≈ noise. For −2.0 ms to be real on *your* config, the fork's baseline must have differed (no fused tail, or no temporal batching) — and DS4's fused tail plus the represented Q side of `69a5e83` likely already contain part of that claim. **Double-count risk.** Max 2.0 (unverified), likely 0.3–0.8. Memory 0. TP: rank-local; note kernel A's per-block RMS recompute raises register/LDS pressure on the projection — the oracle must measure net kernel time, not launch savings. Smallest oracle: extend `scripts/f16_hc_five_row_bench.cu` to 3-chain vs 2-chain at exact shapes, bitwise + timing.
3. **AMD GEMV epilogue fusion (PR 67 pattern).** Boundary: MMVQ/GEMV epilogue ← elementwise activation/mul/residual-view; order-preserving elementwise is fingerprint-safe. DS4 already ships fused mid and fused tail, so the residual population is smaller than llama.cpp's 1263. Max ~0.3–0.4, likely 0.1–0.25. Caveat: any tail that crosses the TP combine boundary is out of scope. Smallest oracle is free: a launch census of sub-5 µs kernels and inter-kernel gaps from the existing `profile/rocprof-step12`/`41934_results.db` — no new run, and it simultaneously bounds candidates 2–5.
4. **KV post fusion (`69a5e83` KV side: RoPE+FP8+store).** Verified: `fp8_kv_quantize_kernel` is still a separate one-launch step. One launch/layer + one tiny round-trip: max ~0.15, likely ≤0.1. Exact iff FP8 rounding identical. Do only as a rider on #2's kernel work. Same bucket: un-stubbing the `fd71740` compressor pair+store — but temporal batching bypasses it "often", so its residual is boundary-token-only, likely <0.1.
5. **Replacement-layout K/V fusion (PR 59 analogue).** PR 59 was **flat on large models** by its own report, and the occupancy win (12.8→16 waves) only exists for small-N dispatches; DS4's F16 projections are already paired (`matmul_f16_pair_tensor` throughout the decode path). A replacement layout also skirts the cache rule: unless the repack is a true bit-exact resident *replacement* (the `e221241` precedent), it is a duplicated resident repack, which is on your excluded list. Max 0.2, likely ~0. Open it only if the census in #3 finds unpaired N≤1024 F16 GEMVs.

**Numerical-correction candidates**

6. **AITER QK norm+RoPE+quant (PR 3320) / AITER compressor attention.** GPT-J RoPE flavor, gfx950 wave64 asm, TP=8 shapes: not exact-portable, and the fusion *boundary* it demonstrates is exactly candidates #2/#4, which you can do exactly. It contributes no independent ms. Reject as a mechanism; keep only as boundary confirmation. Anything that regroups reductions (dynamic expert reassignment, split-wait halves, Q-B-style fusion) stays closed — `45d1f85987f49e86` is the standing counterexample.

### Gaps that change the decision

- **Sum check: the exact-fusion stack (#2+#3+#4+#5) likely totals 0.5–1.2 ms, max ~2.5 ms against a 4.14 ms gap.** Without #1 there is no arithmetic path to 21 t/s. Therefore sequencing HC fusion first buys the least information per day; the ordering probe decides whether 21 is reachable at all.
- **Mechanism I is recorded as "5/7 bitwise PASS" in STATUS.md.** Before treating row sharding as exact-arithmetic, name the two non-bitwise cases and why they don't touch the production path; otherwise #1 silently becomes a numerical-correction candidate and the fingerprint gate will catch it late.
- **Transport asymmetry.** OdinLink baseline is 18.89, so 21 on RoCE still leaves OdinLink ~2 t/s short; and the ordering probe's guarantees are per-transport. Budget both from the start or state that 21 is RoCE-only.
- **The −2.0 ms HC claim is unverifiable from this tree** (commit message only; the fork's baseline config is not in the packet). Treat it as an upper bound to falsify with the microbench, not a planning number.

### Highest-information next experiment

Two cheap, independent steps, in parallel: (a) **the two-transport completion-word ordering probe** — it gates the only ≥2 ms pool and hence the whole campaign; (b) **the launch census from the already-captured rocprof database** — zero-risk, and it collectively bounds mechanisms #2–#5 before any kernel is written. If the census bounds the fusion pool below ~1.5 ms and the ordering probe fails on either transport, the correct output of Round 1 is that 21 t/s is closed under current constraints, and the record should say so explicitly.
