# Producer-ready / overlapped FFN gate exchange (ROCm TP=2 over OdinLink)

## Executive summary

**Recommendation: no-go for full per-row producer-ready streaming in the current optimization scope. Pursue only a conservative, chunked prefill experiment after adding an explicit split publish/wait gate API, and do not pursue decode producer streaming.**

The prerequisite named in the proposal, `ODL_VERBS_DIRECT_SEND=1`, removes the provider's userspace bounce copy and lets `ibv_post_send` queue a registered-buffer address (`OdinLink-Five/verbs/src/odl_tb5_verbs_qp.c:481-518,568-605`). It does **not** establish that a GPU row is finished, and it requires the send buffer to remain unchanged until its send completion (`OdinLink-Five/verbs/src/odl_tb5_verbs_qp.c:489-505`). DS4 currently publishes an FFN gate only after all preceding null-stream compute, by placing `hipStreamWriteValue64` on a blocking gate stream and immediately placing a release wait behind it (`ds4_rocm.cu:712-755`). The service thread then calls the entire synchronous exchange and releases the GPU only when it returns (`ds4_rocm.cu:370-415`; engine callback at `ds4.c:56626-56641`).

There is no per-row or per-owned-expert readiness object in the routed-MoE implementation. Decode's terminal kernel computes all six routed slots for each output coordinate and performs one final assignment (`rocm/ds4_rocm_moe.cuh:2065-2092`); the ordinary prefill fallback writes all row sums in one batch kernel (`rocm/ds4_rocm_moe.cuh:2874-2883`); large sorted prefill can instead zero a whole output matrix and accumulate expert tiles with atomics (`rocm/ds4_rocm_moe_launch.cuh:1687-1695,1723-1747`). None increments a completion counter after all coordinates of a token row are complete. Creating a correct per-row signal therefore requires invasive kernel bookkeeping and a new transport protocol, not merely moving the existing gate call upward.

The smallest defensible experiment is prefill-only, fixed contiguous chunks, with each chunk published only after a GPU event/flag that is ordered after a kernel that has *fully materialized that chunk*. Under the current expert-sorted batch dispatch even that likely requires changing dispatch granularity (or a separate finalization pass); it is not available as a scheduling-only patch. Decode has one row, so “per-row” streaming offers no subdivision: its row gate can begin only after the terminal sum kernel has completed, exactly where the current stream-ordered arrival already places it.

# 1. Exact current mechanism and the readiness problem

## Routing and ownership

The router produces six selected expert IDs and six weights per token: the batch call passes `DS4_N_EXPERT_USED` to `ds4_gpu_router_select_batch_tensor` (`ds4.c:28726-28743`), while decode does the same through `ds4_gpu_glm_router_select_tensor` (`ds4.c:40246-40263`). Under TP=2, ROCm remaps every `(token, slot)` pair into a contiguous local shard. Rank 0 owns the low half and rank 1 the high half, with rank 1 taking an odd remainder (`ds4_rocm.cu:835-859`). Owned IDs become `e-lo`; unowned IDs become local ID zero with weight `0.0f` (`ds4_rocm.cu:835-848`). The launch wrapper shifts all three expert weight bases by the shard offset and reduces `n_total_expert` to the shard size (`rocm/ds4_rocm_moe_launch.cuh:2718-2745` for one row; `:2765-2795` for a batch). The model-residency code independently documents the same low/high physical split (`ds4.c:54335-54352`).

## How a local row becomes complete

There is no host-side loop that adds “owned expert 1, then owned expert 2” into the gate buffer. The ordering and data structure are kernel-specific:

* For the normal one-token/decode Q2 path, `routed_moe_launch` selects `use_direct_down_sum6` when `n_tokens == 1` (`rocm/ds4_rocm_moe_launch.cuh:893-895`). It launches `moe_down_sum6_qwarp32_kernel` (`rocm/ds4_rocm_moe_launch.cuh:1662-1686`). Each output coordinate loops through the six slot indices in order, adds their dot products into a thread-local `total`, and assigns `out[row] = total` once (`rocm/ds4_rocm_moe.cuh:2065-2092`). Zero-weight remapped slots contribute zero because routing weights are applied when `mid_out` is produced (`rocm/ds4_rocm_moe.cuh:2055-2061`). Thus “row fully summed” means **every work item covering all 4096 output coordinates of this single terminal kernel has finished**. There is no expert-completion object to observe.
* For batch/prefill fallback, each expert-slot down result is stored in a three-dimensional logical `down[(token,slot,coordinate)]` scratch, and `moe_sum_kernel` walks the slot dimension in order and assigns the final `out[(token,coordinate)]` (`rocm/ds4_rocm_moe.cuh:2868-2883`). The launch site covers `n_tokens*out_dim` elements (`rocm/ds4_rocm_moe_launch.cuh:1885` and analogous terminal sites). Here “row fully summed” means every coordinate work item belonging to that token in the final sum kernel has completed.
* For large sorted batches, `use_atomic_down` is selected for expert-tiled batches at `n_tokens >= 128` (`rocm/ds4_rocm_moe_launch.cuh:883-886`). The launcher first zeros the entire output matrix (`rocm/ds4_rocm_moe_launch.cuh:1687-1690`), then expert-tile kernels atomically add into token rows; both hot and cold Q4K paths can target the same `out` matrix (`rocm/ds4_rocm_moe_launch.cuh:1723-1747`), and corresponding kernels contain `atomicAdd` sites (`rocm/ds4_rocm_moe.cuh:2366,2429,2556,2619,2689,2761,2835`). Here the row is complete only after **all tiles which can target that token and every coordinate range within them** have finished. The existing sort metadata (`sorted_counts`, `sorted_offsets`, tile descriptors used at `rocm/ds4_rocm_moe_launch.cuh:1694-1700`) schedules work; it is not a row-ready counter.

The application-level prefill path calls the complete batch routed-MoE dispatch and only afterward encodes the bulk FFN gate (`ds4.c:29052-29081,29127-29137`). The gate exchanges the entire `n_tokens * DS4_N_EMBD * sizeof(float)` matrix and the following kernel adds peer and local matrices in canonical rank order (`ds4.c:29131-29144`). Decode similarly dispatches the one-row routed MoE and then calls the row gate before combining local and peer rows (`ds4.c:40442-40466`; the other DS4 decode path has the same ordering at `ds4.c:23968-23988`).

## Existing readiness and why it is not reusable per row

`ds4_tp_encode` records one request in a 1024-entry channel ring, issues one `hipStreamWriteValue64` arrival, then issues one `hipStreamWaitValue64` release wait (`ds4_rocm.cu:712-755`). The big request snapshots only `(out_ptr,in_ptr,bytes)` (`ds4_rocm.cu:768-787`). The service thread observes the monotonic arrival flag, calls the row/batch/big exchange, then release-stores the sequence (`ds4_rocm.cu:370-415`). This is a reusable **whole-gate, whole-prior-stream** readiness signal, not a row signal.

No per-token completion array, per-token atomic counter, row event, or kernel-written row flag exists in the located ROCm routed-MoE launch or kernels. The atomics above update output values, not completion state. Consequently, per-row producer readiness needs new bookkeeping from scratch. Kernel completion as a whole is already represented by stream ordering, but it cannot distinguish row 47 from row 48.

# 2. Concrete meaning for prefill big-gate and decode row-gate

## Prefill: batched big gate

Today prefill supplies one descriptor for the entire matrix (`ds4.c:29131-29137`). `ds4_tp_big_gate_exchange` first performs a TCP header rendezvous and then enters the bulk RDMA exchange (`ds4_tp.c:1780-1799`). The RDMA routine splits bytes into bulk chunks, posts a linked receive list and send list, and then polls until **every** send and receive in that round completes (`ds4_tp.c:1249-1337,1341-1398`). Only after the function returns does the service thread release the GPU.

“Start during compute” cannot mean posting this existing whole-matrix send early: with direct send, OdinLink retains the caller buffer address until the worker/kernel copies it, and DS4 must not modify it before the send completion (`OdinLink-Five/verbs/src/odl_tb5_verbs_qp.c:489-505`). Posting while later GPU kernels still write any covered row is a data race.

A concrete prefill implementation would instead need all of the following:

1. Partition the matrix on row boundaries into deterministic contiguous chunks identical on both ranks. Each request must describe only fully produced rows, not the whole allocation.
2. Produce or finalize chunk 0; publish a chunk-0 arrival; post only chunk 0's receive/send. Continue producing a disjoint chunk 1 while the provider owns chunk 0's send buffer.
3. Give every chunk a distinct transport sequence/header and completion/release state. Current `DS4_TP_BIG` describes one pointer and one byte count and the transport header has only the layer/tag/sequence, not row base/count (`ds4_rocm.cu:768-787`; `ds4_tp.c:1780-1793`). This is a protocol change.
4. Combine each received chunk in canonical rank order, but do not expose the layer's next HC state until every chunk has combined. Current canonical ordering is explicit at `ds4.c:29139-29144`.

The structural obstacle is upstream of the transport: expert-sorted prefill schedules expert tiles across tokens, and for large batches multiple kernels can atomically target the same row. Contiguous row chunks are not known complete merely because an early grid region finished. Safe chunk production likely requires either (a) dispatching routed MoE independently per row chunk, sacrificing global expert sorting/reuse, or (b) retaining the global expert dispatch and adding an explicit per-row finalizer/readiness scheme after all contributing tiles. Option (b) gives little compute overlap if finalization still waits for all expert kernels.

A less ambitious and structurally safer overlap, already described in prior local research, is to exchange finished chunks while the GPU performs the **dependent combine/HC work on earlier received chunks**. It overlaps transport with post-gate compute, not with routed-MoE production. That still requires splitting the current encode into publish and wait operations, because `ds4_tp_encode` immediately enqueues its wait (`ds4_rocm.cu:733-755`), but it avoids guessing that a routed row is ready.

## Decode: one per-token row gate

Decode has exactly one 4096-float row. The local row is not incrementally publishable at expert granularity: `moe_down_sum6_qwarp32_kernel` loops over all six slots for every output coordinate and writes the final coordinate once (`rocm/ds4_rocm_moe.cuh:2065-2092`). Publishing after “three owned experts” is not expressible in the current kernel; there is no intermediate row buffer representing that prefix.

The current row gate is already placed immediately after routed dispatch (`ds4.c:40442-40466`). Moving the host call textually earlier would not start it earlier because `hipStreamWriteValue64` is ordered after null-stream compute by the blocking gate stream; weakening that ordering would permit partial reads. Subdividing the vector by coordinate columns would require multiple transport messages and combines for a 16 KiB row, increasing fixed latency and sequence pressure; it would not overlap with useful independent decode work because the following FFN combine/HC path depends on the whole row. Therefore decode producer-ready overlap is a **no-go** absent a fused transport-aware down kernel and a substantially different collective protocol.

# 3. Core correctness hazard and required readiness contract

The DwarfStar Flash shape has 256 experts (`ds4.c:535-550`). Construct token row 47 with selected global IDs `{3, 19, 144, 161, 172, 188}` in the router's six-slot order. The real TP split assigns `[0,128)` to rank 0 and `[128,256)` to rank 1 (`ds4_rocm.cu:851-859`). Remapping therefore leaves rank 0 with two nonzero weighted slots and four zero-weight slots, and rank 1 with four nonzero weighted slots and two zero-weight slots (`ds4_rocm.cu:835-848`).

On rank 1's large prefill atomic path, suppose expert 144's tile finishes quickly and atomically adds its contribution into `out[47,*]`, while expert 188's tile is delayed by cache behavior or an injected delay. If a proposed CPU readiness poll treats “row 47 changed,” “first owned expert finished,” or even “three of four expert tiles launched” as ready, it can call `ibv_post_send` for row 47. With `ODL_VERBS_DIRECT_SEND=1`, the provider queues the actual registered address rather than making the old immediate userspace bounce copy (`OdinLink-Five/verbs/src/odl_tb5_verbs_qp.c:481-518,584-605`). The provider/kernel may copy the row before expert 188's `atomicAdd`s land. Rank 0 then receives

`partial_rank1 = contribution(144) + contribution(161) + contribution(172)`

instead of the required

`partial_rank1 = contribution(144) + contribution(161) + contribution(172) + contribution(188)`.

Both ranks subsequently perform the ordinary local/peer float add (`ds4.c:29139-29144`). No bounds error, timeout, CQ error, or crash occurs. The missing expert changes row 47's FFN result and all dependent layers: silent wrong output.

The converse race is also possible: the provider begins copying a 16 KiB row while a late GPU workgroup updates later cache lines. The wire image can then contain a torn mixture of old and new coordinates even if some scalar flag claims readiness. A CPU release fence before `ibv_post_send` (`ds4_tp.c:1297-1304`) orders CPU operations; it cannot retroactively complete unrelated GPU writes.

A correct readiness contract must guarantee:

* The expected count is computed per token as the number of selected slots whose global expert belongs to this rank. In the example it is 2 on rank 0 and 4 on rank 1. Counting a zero-weight remapped placeholder as an owned completion would be wrong.
* A completion cannot be credited merely when an expert tile starts or finishes one coordinate tile. It must mean that expert's entire contribution to every coordinate of that row is globally visible, or that a finalizer has materialized the full row.
* If several kernels contribute, the last contributor must perform a device-scope release operation after its payload writes; the observer must acquire the flag. A plain atomic increment without release ordering is insufficient. The current measured whole-stream mechanism uses `hipStreamWriteValue64` after production and a host atomic acquire (`ds4_rocm.cu:733-750,370-375`), and its source comment explicitly limits the payload-visibility measurement to one producing kernel on one stream (`ds4_rocm.cu:142-177`). A new multi-kernel/per-row design requires a new hardware probe on gfx1151 proving payload visibility for that exact pattern.
* The CPU may post only the exact row/chunk whose release-acquire handshake has succeeded. It must not post a larger SGE containing unready rows.
* The GPU must not reuse or further modify a posted send region until its matching send completion, per direct-send's buffer lifetime contract (`OdinLink-Five/verbs/src/odl_tb5_verbs_qp.c:489-505`). The receive region similarly cannot be consumed until receive completion and the existing acquire fence (`ds4_tp.c:1398-1405`).
* Both peers must agree on chunk identity and order. Current big-gate headers validate layer/tag/sequence only (`ds4_tp.c:1780-1793`); streaming needs row range or a deterministic derivation that is validated on both sides.

**Conclusion:** full per-row streaming readiness is technically achievable, but not without a much bigger redesign: kernel-level completion accounting, device memory-order proof, chunked protocol state, buffer lifetime management, and likely changes to expert-sorted dispatch. It is not safely achievable as a small extension of direct-send. The safer partial variant is (1) transport/post-combine chunk pipelining after routed-MoE has globally completed, or (2) at most a conservative last-chunk experiment in which an explicit kernel/event proves the earlier chunk immutable. A time-based “margin” alone is not a correctness mechanism and must never be production logic.

# 4. Validation plan for a correctness-risk change

The acceptance criterion is **bit-exact tensor output versus the same build and routing baseline with overlap disabled**, not merely the same sampled text and not a tolerance-based accuracy test. Chunking must preserve the six-slot summation order inside each rank and the rank-0-then-rank-1 combine order already used at `ds4.c:29139-29144`; changing floating-point association is outside this proposal.

1. Add a diagnostic mode (in the eventual implementation, not in this planning task) that dumps router selected IDs/weights, each rank's local partial immediately before publication, received peer partial, and post-combine output for chosen layers/rows. Compare every 32-bit float bit pattern against non-overlapped baseline.
2. Test prompt sizes `1`, `chunk-1`, `chunk`, `chunk+1`, multiple chunks, maximum configured prefill, and non-divisible tails. Exercise both below and above the atomic-down threshold of 128 tokens established at `rocm/ds4_rocm_moe_launch.cuh:883-886`.
3. Generate adversarial routing tables with the same Zipf/skew pattern used for this session's WMMA work: all rows selecting the same six experts; hot experts split 3/3 across ranks; 1/5 and 5/1 ownership; all six on rank 0; all six on rank 1; alternating low/high IDs; and rows whose expected owned count is zero. Include duplicate-selection cases if the router can emit them; otherwise assert uniqueness and test that invariant.
4. Add test-only late-expert delay injection inside a selected expert/tile path, keyed by `(layer, expert, token row)`. Delay the final owned expert for row 47 while allowing subsequent rows/other experts to advance. Sweep delay placement before the payload store, between coordinate tiles, and immediately before readiness publication. This must make an intentionally broken early-publish build fail reliably, proving the test can detect the race.
5. Run thousands of exchanges with randomized schedules, prompts, seeds, batch sizes, chunk sizes, and both ranks alternately slowed. Poison output and receive buffers before every layer so stale rows are conspicuous rather than accidentally correct zeros.
6. Compare overlap-off/on full logits and relevant intermediate tensors byte-for-byte for every token. Sampled token/text equality is a secondary smoke check only. Run temperature-zero and stochastic decoding with recorded RNG state; routing and tensors must match before sampling.
7. Run ThreadSanitizer/ASan where host-side code permits, provider stress/CQ checks, and sequence/header fault tests. Require no gate desync, CQ error, timeout, missing/duplicate chunk, or use-after-lifetime. The gate request currently snapshots device pointers because tensor view descriptors may be freed immediately (`ds4_rocm.cu:780-786`); chunk views must preserve the backing allocation until send and receive completions.
8. Repeat the existing GPU payload-visibility probe in the new exact topology: multiple producer kernels/streams, device release counter, CPU acquire, direct RDMA read. The existing comment explicitly warns not to generalize its one-kernel/one-stream result (`ds4_rocm.cu:142-177`).
9. Only after bit-exact stress passes, benchmark with and without `ODL_VERBS_DIRECT_SEND=1` to separate scheduling benefit from provider-copy benefit. A failure is correctness-blocking; there is no acceptable small mismatch rate.

# 5. Implementation sketch and staging

## Stage 0: measurement and invariants, no overlap

Instrument kernel completion, service-thread notice, post-send return, last send/receive CQE, combine completion, and HC completion per layer/chunk. Confirm which routed-MoE launch branch is active at each prompt size. Record the selected IDs and expected owned-slot count. This prevents designing around the wrong terminal kernel.

## Stage 1: split gate publication from waiting, behavior unchanged

Refactor conceptually (when implementation is authorized) so a gate can be reserved/published and later waited, while initially calling publish+wait back-to-back. Preserve channel sequence advancement, ring overflow handling, failure release, and pointer snapshot semantics from `ds4_tp_encode` (`ds4_rocm.cu:712-755,768-787`). Require bit-exact output and identical transport order before introducing concurrency.

## Stage 2: safest useful overlap—prefill transport with post-gate work

Use fixed row-aligned chunks (start conservatively at 128 or 256 rows), but publish them only after the entire routed-MoE batch has completed. Pipeline exchange of chunk `k+1` with canonical combine and HC expansion of already received chunk `k`. Do not allow next-layer work or buffer reuse until every chunk is complete. This does not realize producer-ready routed-MoE overlap, but it is useful if combine/HC work is measurable and exercises the chunk protocol without the highest-risk readiness problem.

This stage requires the transport to accept row-offset/byte-range requests and multiple in-flight big chunks. It must continue posting receives before sends as the present bulk routine does (`ds4_tp.c:1274-1333`) and must retain buffers through completions.

## Stage 3: conservative producer overlap experiment, prefill only

Do **not** use a fixed time margin. Instead, choose a dispatch boundary that provides a formal completion event. The smallest candidate is two contiguous row batches: launch/finalize the first `N-chunk` rows, publish them, then launch the final chunk while the first exchange proceeds. This deliberately sacrifices some global expert batching to obtain provable disjointness. Benchmark the lost routed-MoE efficiency against communication saved. Start with only one early prefix and one late tail, default off, on exactly TP=2/gfx1151/direct registered slab/no directional steering.

If two-batch dispatch loses more compute than it hides, stop. Do not proceed to counters.

## Stage 4: full per-row readiness only as a separately approved redesign

If measurements justify it, design per-row expected/completed state, device-release publication, CPU-acquire consumption, row-range transport descriptors, and generation counters preventing ABA/reuse. For atomic expert tiles, a robust pattern is likely a separate row-finalization kernel launched only after all down contributors, rather than asking arbitrary coordinate workgroups to elect a last writer. But because such a finalizer follows the contributing grid, it may expose rows only at whole-kernel granularity and erase the hoped-for producer overlap. This stage needs its own design review and is not recommended now.

Decode is excluded from Stages 2-4. Its one-row terminal sum provides no safe/useful row pipeline (`rocm/ds4_rocm_moe.cuh:2065-2092`).

# 6. Expected payoff, honestly bounded

The session's earlier profiling attributes about 39% of layer time to the FFN gate. That is an **upper bound**, not an expected saving. Direct-send already removes the provider bounce copy, while DS4's direct big-gate path avoids staging when both buffers fall inside the registered slab (`ds4_tp.c:1239-1247`); it posts all WRs and still waits for all send/receive completions (`ds4_tp.c:1297-1398`). Therefore the remaining gate bucket is primarily fixed rendezvous, posting/progress, wire/DMA, peer skew, and completion wait. `ODL_VERBS_DIRECT_SEND=1` makes earlier posting possible but also lengthens the period during which the GPU must not touch the posted buffer.

For any overlap window `C` and remaining gate time `G`, the theoretical layer saving is at most `min(C,G)`, minus chunk protocol overhead, lost MoE batching efficiency, extra flags/atomics, additional headers/CQEs, and exposed tail imbalance. The 39% figure bounds `G/layer_time`; it does not show that 39% can be hidden.

* **Decode:** expected producer-ready saving is approximately zero. There is one row and its gate can start only when the one terminal sum kernel finishes. Splitting 4096 coordinates creates more small messages and no substantial independent dependent work.
* **Prefill Stage 2 (post-gate pipeline):** likely modest. It can hide transport only behind chunk-local combine and HC expansion, whose duration must be measured. If those kernels are short relative to the fixed round trip, most of the gate remains exposed.
* **Prefill two-batch producer experiment:** plausible benefit ranges from negative to a fraction of the 39% gate share. The maximum overlap is the routed-MoE compute time of the deliberately delayed tail. If the last chunk is small, it cannot hide much fixed gate latency; if it is large, the first exchange starts later and splitting may degrade expert sorting/cache reuse. A reasonable go/no-go target is at least a repeatable 5% end-to-end prefill improvement after including the dispatch-efficiency loss; below that, the correctness and maintenance cost is unjustified.
* **Full per-row streaming:** no defensible numeric estimate exists without a trace of actual row/tile completion and transport progress. The present batch API structurally requires a whole pointer/byte range and a synchronous all-completion return (`ds4_rocm.cu:768-787`; `ds4_tp.c:1210-1410`). If rows cannot be dispatched independently without harming MoE throughput, the overlap is structurally unavailable in the current batching design.

Bottom line: direct-send is a necessary transport optimization but not a producer-readiness primitive. Approve Stage 0-2 if the goal is low-risk learning and possibly modest prefill gain. Approve Stage 3 only behind strict bit-exact gates and only if measurement shows enough routed-MoE tail to cover meaningful transport. Do not approve full per-row readiness or any decode version on the present evidence.
