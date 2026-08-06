# FFN gate F32 wire-compression design for TP=2 OdinLink RDMA

## 1. Exact mechanism

The candidate applies only to the routed-MoE FFN partial exchanged between the two TP ranks. It does not apply to attention gates, router logits or weights, shared-expert tensors, producer-ready overlap, or the final residual stream.

### Model and exchange frequency

The relevant GLM 5.2 shape has 79 layers, hidden width 6,144, three leading dense layers, and one trailing next-token-prediction layer (`ds4.c:611-639`). Normal decode therefore executes one FFN exchange in each of:

\[
79 - 3 - 1 = 75
\]

sparse layers per generated token. The schedule explicitly skips the three dense layers and the trailing MTP layer (`ds4.c:56575-56589`). This is substantially more than the “40+ layers” minimum in the risk statement.

### Decode production and consumption

For a normal sparse decode layer:

1. The rank-local routed-MoE dispatch writes its owned-expert partial directly into the layer’s FFN TP output slab view. The destination selection is at `ds4.c:40427-40435`; the routed dispatch producing it is at `ds4.c:40442-40459`.

2. `ds4_gpu_tp_gate_encode(il, DS4_TP_GATE_FFN)` enqueues the FFN gate (`ds4.c:40461-40462`). Gate 1 is explicitly the FFN gate, while gate 0 is attention (`ds4_tp.h:29-35`).

3. The GPU TP runtime records the row request (`ds4_rocm.cu:758-760`). Its service thread invokes the registered exchange callback for row gates (`ds4_rocm.cu:372-406`), which reaches `ds4_engine_tp_exchange()` and then `ds4_tp_gate_exchange()` (`ds4.c:56610-56615`).

4. On RDMA, `tp_rdma_gate_exchange()` sends the local slab output slot and receives the peer into the matching input slot. The send address is calculated at `ds4_tp.c:1032-1049`; the SEND work requests cover `tp->vec_bytes` at `ds4_tp.c:1057-1078`; posted receives land in the input slot at `ds4_tp.c:992-1027`.

5. Once the gate releases, the canonical combine is an F32 elementwise add of local and peer partials:

   ```text
   ffn_out = tp_out[slot] + tp_in[slot]
   ```

   at `ds4.c:40461-40466`. The ROCm add implementation requires all three tensors to be F32 and executes `float` addition (`rocm/ds4_rocm_misc_launch.cuh:1-5`).

The local partial is a sum over the routed experts owned by that rank. Expert ownership is established by rebasing selected experts onto a contiguous rank-local shard (`ds4_rocm.cu:851-900`), and the routed-MoE launcher writes an F32 output of `out_dim` elements (`rocm/ds4_rocm_moe_launch.cuh:2698-2749`). The code’s correctness argument is that rank 0’s and rank 1’s expert-weighted partial sums reconstruct the full routed result when added (`ds4_rocm.cu:821-827`).

### Prefill production and consumption

For multi-token prefill:

1. Sparse-layer batch routing and MoE execution are entered after router selection (`ds4.c:41946-42016`).

2. Under TP=2, `glm_graph_tp_batch_bounce_ready()` allocates one local and one peer tensor sized as:

   \[
   n_\text{tokens} \times 6144 \times \text{sizeof(float)}
   \]

   (`ds4.c:39851-39860`, `ds4.c:42000-42003`).

3. The rank-local batch routed-MoE dispatch writes its partial into `tp_bounce_out`; `glm_graph_tp_batch_ffn_combine()` passes that tensor and `tp_bounce_in` to `ds4_gpu_tp_big_gate_encode()` with the exact byte count (`ds4.c:39844-39868`, `ds4.c:39878-39880`).

4. The big-gate encoder snapshots the two device pointers and byte count into the asynchronous request (`ds4_rocm.cu:768-787`). The service thread invokes `g_tp_big_fn` for that request (`ds4_rocm.cu:398-404`), eventually reaching `ds4_tp_big_gate_exchange()` through `ds4_engine_tp_big_exchange()` (`ds4.c:56626-56641`).

5. The RDMA bulk path exchanges the complete byte range via 128 KiB SEND/RECV chunks (`ds4_tp.c:1202-1213`, `ds4_tp.c:1249-1287`, `ds4_tp.c:1300-1333`). When the caller’s tensors are outside the registered slab, the path stages each outgoing chunk into the slab at `ds4_tp.c:1263-1271`.

6. After the gate, prefill reconstructs the full routed output with the same F32 local-plus-peer add (`ds4.c:39883-39886`).

### Exact wire tensor

For the GLM 5.2 model, `DS4_N_EMBD` is 6,144 (`ds4.c:615-617`). `vec_bytes` is defined as `n_embd * sizeof(float)` when the TP engine is bound (`ds4.c:56670-56672`) and in the slab sizing calculation (`ds4_tp.c:614-623`).

Therefore:

| Mode | Logical shape per FFN gate | Current dtype | Elements | Current bytes in each direction |
|---|---:|---:|---:|---:|
| Decode | `[6144]` | F32 | 6,144 | 24,576 B = 24 KiB |
| Prefill/batch | `[rows, 6144]` | F32 | `rows × 6,144` | `rows × 24,576 B` |
| Proposed compressed wire | same logical shape | BF16 or F16 | unchanged | `rows × 12,288 B` |

Each rank sends its own partial and receives the peer’s partial, so aggregate full-duplex traffic is twice those per-direction figures. Compression halves both directions.

The source comment independently describes the decode object as a 24 KiB routed-FFN partial (`ds4.c:37501-37504`). This is an all-reduce in mathematical effect, but the implementation is a symmetric exchange followed by a local add on each rank—not an ibverbs collective.

## 2. Compression scheme options

### What the source establishes about the tensor’s range

The tensor contains weighted routed-expert down-projection sums before the canonical cross-rank add. It is not a probability tensor and has no code-enforced narrow interval.

The expert-routing weights are retained when selections are rebased; unowned selections receive zero weight, while owned selections keep their original weights (`ds4_rocm.cu:835-848`). The rank-local routed result is then produced by the MoE launcher in F32 (`rocm/ds4_rocm_moe_launch.cuh:2698-2749`). GLM 5.2’s configured SwiGLU clamp is zero, meaning this shape does not provide a finite activation bound through an enabled clamp (`ds4.c:639-647`).

No checked source file contains measured min/max, percentile, exponent-histogram, NaN, or infinity statistics for this exact `tp_bounce_out`/`tp_out` tensor. Consequently, the repository establishes the tensor’s semantics and dtype, but not its empirical numeric range. Claiming that it is always within F16 range would be guessing.

A prerequisite measurement should therefore record, separately for every sparse layer and for prefill and decode:

- minimum and maximum;
- maximum absolute value;
- fraction with `abs(x) > 65,504`;
- fraction below the normal and subnormal F16 ranges;
- BF16 and F16 round-trip absolute and relative errors;
- NaN and infinity counts.

This instrumentation belongs in validation builds, not the eventual fast path.

### BF16

BF16 retains F32’s eight-bit exponent and therefore approximately the same dynamic range. It reduces the significand from F32’s 23 stored fraction bits to seven.

Advantages for this tensor:

- Overflow is not a credible new failure mode for any finite F32 value that is representable in BF16’s range.
- It is robust if occasional routed partials have unexpectedly large magnitude.
- No dynamic scaling or per-row metadata is required.
- Pack and unpack remain a fixed two-byte-per-element operation.

Disadvantage:

- Around ordinary activation magnitudes, BF16 has coarser relative resolution than F16. Its unit roundoff is approximately \(2^{-8}\), versus \(2^{-11}\) for F16 under round-to-nearest.
- Repeating this conversion at 75 sparse layers per normal token makes that loss important even though every individual conversion is small.

### F16

F16 has a ten-bit fraction and therefore materially better relative precision for values in its normal range, but only a five-bit exponent. Its largest finite value is 65,504.

Advantages:

- If the measured partials remain comfortably within F16 range, F16 should introduce substantially less rounding error than BF16.
- ROCm already includes F16 conversion machinery, including `f32_to_f16_kernel` uses in the MoE paths (`rocm/ds4_rocm_moe_launch.cuh:1177-1187`), so the representation is operationally familiar.
- It still halves the wire bytes.

Risks:

- A value above 65,504 can become infinity unless the implementation explicitly clamps or scales it.
- Clamping would introduce a larger, biased error and should not be silently added.
- Dynamic per-row scaling would change the design into a quantization codec with metadata and additional kernels; that is outside the simple BF16/F16 proposal.

### Recommendation

F16 is the likely accuracy winner only if measurement proves a comfortable range margin. The proposed selection rule is:

- Choose F16 only if the complete validation corpus has no non-finite values, no values with `abs(x) > 32,752`—a 2× safety margin below F16 overflow—and no concerning layer-wise tail growth.
- Otherwise choose BF16.
- Do not clamp to make F16 pass.
- If the range measurement is unavailable, begin the implementation experiment with BF16 because it is safer against catastrophic overflow, but do not infer that BF16 will meet the accuracy criterion.

This is intentionally conditional. The real tensor range has not yet been measured in the checked source tree.

### llama.cpp precedent

The requested file exists at `/home/wkljohn/Desktop/cc/llama.cpp-tq3-hip-new/ggml/src/ggml-cuda/allreduce.cu`.

Its chunked all-reduce explicitly supports a destination type different from the wire type, including F32 destination with BF16 wire, to halve PCIe bytes (`llama.cpp-tq3-hip-new/ggml/src/ggml-cuda/allreduce.cu:79-99`). It performs an ordinary cast from the destination type to the wire type (`llama.cpp-tq3-hip-new/ggml/src/ggml-cuda/allreduce.cu:133-146`). On reduction it rounds both the local and peer operands through the same wire type before adding, specifically so both GPUs produce bit-equivalent results (`llama.cpp-tq3-hip-new/ggml/src/ggml-cuda/allreduce.cu:95-99`, `llama.cpp-tq3-hip-new/ggml/src/ggml-cuda/allreduce.cu:174-178`).

Thus the precedent is deterministic type conversion, not stochastic rounding. Its “truncate identically” wording describes both operands being narrowed consistently; the actual source uses the project’s typed cast helper rather than hand-masking F32 bits.

That symmetric local rounding is relevant here. If DS4 decompresses only the peer operand while retaining its own full-F32 partial, rank 0 computes `F32(rank0) + rounded(rank1)` while rank 1 computes `rounded(rank0) + F32(rank1)`. The results need not match. Both operands should therefore pass through the selected wire representation before the canonical add.

### AMD QuickReduce precedent

A local AMD QuickReduce implementation exists under `/home/wkljohn/Desktop/cc/vllm-work/vllm/csrc/quickreduce`.

It is not a BF16/F16 narrowing codec for F32 inputs. Its public wrapper accepts Half or BF16 tensors and dispatches either an unquantized F16/BF16 codec or block-scaled INT8, INT6, or INT4 codecs (`vllm/csrc/custom_quickreduce.cu:58-84`, `vllm/csrc/quickreduce/quick_reduce.h:62-67`, `vllm/csrc/quickreduce/quick_reduce.h:164-190`).

For its integer codecs, it:

- computes a block absolute maximum;
- derives a symmetric decoding and reciprocal encoding scale;
- clamps values to the integer codec’s range;
- uses deterministic `rintf` rounding to the nearest integer;
- stores one scale per block.

The Q4 implementation shows this directly (`vllm/csrc/quickreduce/quick_reduce_impl.cuh:57-66`, `vllm/csrc/quickreduce/quick_reduce_impl.cuh:105-155`); Q6 follows the same method (`vllm/csrc/quickreduce/quick_reduce_impl.cuh:258-320`). No stochastic-rounding path appears in the local QuickReduce source. Its optional BF16-to-Half conversion uses explicit round-to-nearest conversion (`vllm/csrc/quickreduce/quick_reduce_impl.cuh:572-583`).

QuickReduce therefore provides precedent for deterministic round-to-nearest, block scaling, and lower-bit symmetric quantization—not for stochastic rounding or raw F32-to-BF16 truncation. Its INT4/6/8 design is materially more complex and more lossy than this proposal, so it should be read for codec structure but not copied into the first DS4 experiment.

## 3. Concrete proposed accuracy tolerance

### Proposed approval criterion

Approve implementation work only if the user accepts this ship criterion in advance:

> Across at least 256 held-out production-representative prompts and at least 131,072 scored next-token positions, spanning short decode, long decode, and long-prefill cases, the compressed path must achieve at least 99.99% greedy top-1 agreement with the current F32-wire TP=2 baseline; maximum absolute logit delta at every scored position must be no greater than 0.05; the 99.99th-percentile absolute logit delta must be no greater than 0.01; and neither path may produce a new NaN or infinity.

All parts are required. A pass on average logit error cannot excuse token changes, and a high token-match rate cannot excuse rare large logit corruption.

The corpus should include:

- source code, prose, structured data, multilingual text, math, and tool-like syntax;
- prefill lengths distributed across the operating range, including the largest supported production chunk;
- at least 32 prompts with 1,024-token greedy continuations;
- adversarial low-margin positions, selected independently from baseline top-1/top-2 margins;
- the same model, weights, prompt template, sampling mode, and TP rank assignment for baseline and candidate.

For 131,072 positions, 99.99% permits at most 13 top-1 mismatches. That is strict enough to detect systematic drift while acknowledging that a deliberate precision reduction cannot reasonably promise exact token identity forever.

A secondary quality guard should compare token cross-entropy under teacher forcing and require no more than a 0.01% relative increase over the F32-wire baseline. This is not a substitute for the primary criterion; it helps detect broad probability degradation that top-1 agreement can miss.

### Why compounding requires a strict end-to-end tolerance

This is not a one-shot storage conversion. Normal GLM decode crosses 75 sparse-layer FFN gates per token (`ds4.c:56583-56589`), and prefill performs the corresponding batch exchange at every sparse layer. Every rounded partial is injected into the hidden-state evolution consumed by later normalization, attention, routing, and MoE decisions.

The errors are not guaranteed simply to add linearly:

- residual connections may damp some errors;
- normalization may limit magnitude growth;
- routing and top-k selection can turn a small hidden-state perturbation into a discrete expert-selection change;
- a changed generated token changes every subsequent model input, causing trajectory divergence even if the initiating logit delta was small.

For that reason, a tolerance based only on the immediate post-gate tensor would be too weak. The proposed logit thresholds are deliberately tighter than a generic “no visible quality regression” check, and validation must include both teacher-forced positions and free-running long generations.

Teacher forcing measures accumulated numerical drift while holding token history fixed. Free-running generation tests the actual compounding failure mode. The 0.05 worst-case logit bound and 99.99% top-1 threshold are proposals, not established facts; the user can approve, tighten, or reject them before implementation begins.

## 4. Validation plan

### The existing bar cannot be met

The project’s established correctness bar is exact bit-identical output diffing. This change cannot meet that bar by design: narrowing any non-exactly-representable F32 partial to BF16 or F16 changes bits before the canonical add. Even if the final token often remains unchanged, intermediate hidden states and logits will normally differ.

The correct classification is therefore “controlled approximate mode,” not “correctness-preserving optimization.” It must remain opt-in unless and until the project explicitly adopts a broader default accuracy policy.

### Required validation methodology

1. **Freeze a baseline.** Use the current TP=2 F32-wire build and fixed model artifact, runtime flags, prompt rendering, seeds, sampling mode, rank assignment, and prefill chunk size.

2. **Collect tensor-range telemetry first.** On the baseline, capture per-layer range and exponent statistics for the exact local tensors passed to the decode row gate and prefill big gate. Use these measurements to apply the F16/BF16 decision rule from section 2.

3. **Validate pack/unpack locally.** For saved real gate tensors, round-trip F32 → candidate wire type → F32 and measure max, percentile, relative, and ULP errors. Confirm deterministic conversion and the expected overflow/subnormal behavior.

4. **Check rank symmetry.** After each combine, optionally hash the resulting F32 tensor on both ranks. Both ranks must match one another bit-for-bit even though neither matches the F32-wire baseline. This requires narrowing both local and peer operands identically.

5. **Teacher-forced end-to-end run.** Score identical next tokens after identical histories, preserving alignment even when candidate top-1 differs. Record:

   - maximum and percentile absolute logit delta;
   - maximum relative delta where meaningful;
   - baseline and candidate top-1/top-k;
   - top-1/top-2 baseline margin;
   - cross-entropy delta;
   - first layer at which hidden-state or router-selection differences become material.

6. **Free-running greedy run.** Generate at least 1,024 tokens for the long-generation subset. Record first divergence, total top-1 agreement before divergence, and whether outputs resynchronize. Do not treat post-divergence token-by-token comparison as a clean numerical metric; the contexts are different by then.

7. **Prefill coverage.** Exercise short and maximum-sized prefill chunks. The prefill tensor is `[rows, 6144]`, so its range distribution and pack cost may differ from single-row decode.

8. **Operational checks.** Require no gate desynchronization, RDMA timeout, new NaN/infinity, receive-buffer visibility failure, or rank mismatch. Run repeated cold and warm sessions and prompt transitions because the big and row gates reuse one QP and explicitly drain the decode receive window before bulk transfers (`ds4_tp.c:1109-1199`).

9. **Performance validation only after accuracy passes.** Measure pack, post-send, completion wait, unpack, combine, whole FFN gate, layer, prefill, and tokens-per-second separately. Existing big-gate profiling already separates staging, posting, and waiting components (`ds4_tp.c:1221-1236`); the new path should add pack/unpack buckets rather than folding them into “wire.”

### Release condition

The feature may ship as opt-in only if:

- all primary accuracy thresholds in section 3 pass;
- both ranks’ post-combine tensors remain bit-identical to one another;
- there are no new transport or numerical failures;
- performance improves on both prefill and decode workloads for which the flag is advertised.

Failure of any condition means the default F32 path remains the only supported path. A BF16 pass does not imply an F16 pass, or vice versa; each wire type needs its own report.

## 5. Implementation sketch (plan only, no code)

### Runtime mode and protocol agreement

Introduce one cached opt-in environment variable, for example:

```text
DS4_TP_FFN_WIRE=bf16
DS4_TP_FFN_WIRE=f16
```

Unset, empty, `0`, or an unrecognized value should retain F32 or fail closed with a clear startup error. There should be no automatic precision choice at runtime based on individual tensors.

Follow the `ODL_VERBS_DIRECT_SEND` convention: default off, parsed once, and enabled only by an explicit recognized value. OdinLink’s implementation caches its opt-in with `pthread_once`, accepts only a leading `1`, and otherwise remains disabled (`OdinLink-Five/verbs/src/odl_tb5_verbs_qp.c:481-518`). DS4 may use an enum rather than a boolean because it has two compressed modes.

The selected wire mode must be added to the TP runtime-feature handshake. The existing hello already compares runtime features and rejects a mismatch (`ds4_tp.c:226-232`), and the identity carries `runtime_features` (`ds4_tp.h:46-66`). Both ranks must agree before inference starts. Never permit one rank to send F16 while the other posts an F32 or BF16 receive.

The kill switch is simply restarting with `DS4_TP_FFN_WIRE` unset. F32 must continue to use the existing allocation, scheduling, RDMA, TCP fallback, and combine paths unchanged.

### Pack and unpack kernels

Add ROCm kernels with these logical operations:

```text
pack_f32_to_bf16(wire_out, local_f32, count)
pack_f32_to_f16(wire_out, local_f32, count)
unpack_bf16_to_f32(peer_f32, wire_in, count)
unpack_f16_to_f32(peer_f32, wire_in, count)
```

Requirements:

- deterministic round-to-nearest conversion;
- vectorized aligned loads/stores where practical;
- explicit handling and diagnostic counters for NaN/infinity;
- no silent F16 clamp;
- bounds-safe support for arbitrary row counts;
- no stochastic rounding in the first implementation.

The unpack destination remains F32 because current consumers require F32 (`rocm/ds4_rocm_misc_launch.cuh:1-5`).

For symmetry, the local operand used in the final add must also be the round-tripped representation. The cleanest layout is:

```text
local producer F32
    -> pack to 16-bit local wire buffer
    -> exchange
    -> unpack local wire buffer to local rounded F32
    -> unpack peer wire buffer to peer rounded F32
    -> existing canonical F32 add
```

An equivalent fused unpack-and-add kernel could later read both wire buffers and produce the F32 sum directly, but it should not be the first version because separate unpacked tensors make validation and rank hashing clearer.

### Decode placement

Today, the decode producer writes directly into the F32 registered slab output view, and RDMA sends that slot (`ds4.c:40427-40435`, `ds4_tp.c:1048-1078`). Compressed mode needs distinct 16-bit wire slots; reinterpreting the existing F32 slot in place would destroy the producer output before symmetric local rounding and complicate rollback.

The TP slab layout should reserve compressed FFN output/input storage only when the flag is active:

- one 16-bit vector per relevant layer for local wire output;
- one 16-bit vector per relevant layer for peer wire input;
- keep or provide F32 local/peer buffers for unpack and the existing add.

The gate stream ordering is the critical integration point. `ds4_tp_encode()` currently queues a GPU-visible arrival write followed by a wait for the service-thread release (`ds4_rocm.cu:712-752`). In compressed mode, the conceptual order must be:

1. producer finishes the F32 local partial;
2. pack kernel writes the 16-bit registered wire slot;
3. GPU gate-ready signal becomes visible;
4. service thread sends the packed slot and receives the peer packed slot;
5. service thread releases the gate;
6. unpack kernels materialize both rounded F32 operands;
7. the existing consumer add runs.

Pack must precede the gate-ready signal; unpack must follow the gate wait. Merely packing in the CPU service thread would reintroduce the slow CPU read of GPU-written memory that this proposal is intended to avoid.

The current code relies on a blocking `g_tp_stream` to preserve gate ordering and warns that the gate must not use the null stream (`ds4_rocm.cu:232-241`). The design should extend that same stream sequence rather than introduce producer-ready overlap. Any cross-stream assumption must be proved with events or existing blocking-stream semantics; this document does not authorize overlap.

`tp_rdma_post_gate_recv()` and `tp_rdma_gate_exchange()` must use a mode-specific wire byte count and mode-specific input/output offsets rather than the global F32 `tp->vec_bytes` (`ds4_tp.c:995-1027`, `ds4_tp.c:1032-1098`). For GLM decode, the wire count becomes 12,288 bytes, fitting in one 16 KiB OdinLink message instead of the current two-message path triggered above 16 KiB (`ds4_tp.c:922-934`). That message-count reduction may provide an additional decode latency benefit beyond byte halving, but it must be measured.

TCP fallback must either support the same negotiated 16-bit payload or reject compressed mode at startup. Supporting it is preferable because the protocol mode is already negotiated and the conversion semantics are transport-independent.

### Prefill placement

Today, `glm_graph_tp_batch_ffn_combine()` passes F32 `tp_bounce_out` and `tp_bounce_in` plus `rows × 6144 × 4` bytes to the big-gate encoder (`ds4.c:39863-39886`). Compressed mode should:

1. retain the routed-MoE producer’s F32 output;
2. allocate or borrow `rows × 6144 × 2` registered wire buffers;
3. enqueue the pack before the big-gate ready signal;
4. pass the packed pointers and compressed byte count through the snapshotted big-gate request;
5. receive the peer’s packed data into the peer wire buffer;
6. enqueue symmetric local and peer unpack after the gate wait;
7. invoke the existing F32 combine.

The request currently snapshots `out_ptr`, `in_ptr`, and `bytes` because view descriptors may be freed before the service thread consumes them (`ds4_rocm.cu:778-787`). Compressed mode must preserve that rule; any extra local-F32 and wire pointers required by post-gate unpack must also be snapshotted or remain graph-owned for the full request lifetime.

Where possible, the 16-bit big-gate buffers should reside directly in the registered slab. That avoids the current non-direct staging path, whose CPU copy reads GPU memory at the measured 200 MB/s (`ds4_tp.c:1217-1269`; `ds4_tp.h:68-79`). However, direct GPU-write/NIC-read visibility has historically been sensitive: the experimental device-copy staging path timed out because GPU DMA writes were not reliably visible to OdinLink (`ds4_rocm.cu:617-633`). Therefore, compressed slab-direct transfer requires its own visibility test. It must not assume that a successful row-gate mechanism automatically proves every bulk allocation and producer path.

This visibility work is ordering, not producer-ready overlap: all packing must complete before the existing gate signal.

### Symmetry requirement

Both ranks must use identical compression, rounding, and canonical operand ordering.

It is technically possible for each rank to send a compressed peer operand while keeping its own partial in F32, but that produces rank-dependent arithmetic:

```text
rank 0: F32(A) + round(B)
rank 1: round(A) + F32(B)
```

Those results can differ. The preferred behavior is:

```text
both ranks: round(A) + round(B)
```

with the same rank-0-then-rank-1 canonical operand order already used elsewhere (`ds4.c:23965-23988`). Asymmetric compression should not be implemented.

## 6. Expected payoff, honestly bounded

### Upper bound for the exchange itself

The payload falls from four to two bytes per element. If transfer time were perfectly proportional to bytes and pack/unpack were free, the payload-dependent portion of each FFN exchange could improve by at most 50%.

That is not a 50% end-to-end inference claim.

For decode, 24 KiB currently exceeds the 16 KiB row-message cap and is split into two messages (`ds4_tp.c:922-934`, `ds4_tp.c:1000-1026`). A 12 KiB compressed vector would fit in one message. The theoretical benefit could therefore include both byte halving and removal of one SEND/RECV pair. Conversely, fixed gate, posting, polling, and synchronization costs may dominate such a small payload, in which case byte halving saves much less than 50%.

For prefill, the current bulk path already uses 128 KiB messages and batches up to 32 chunks per round (`ds4_tp.c:1202-1208`, `ds4_tp.c:1249-1254`). Compression approximately halves chunk and round counts for the same rows, but fixed per-gate barriers and completion processing remain.

### End-to-end bound

Let:

- \(F_d\) be the measured fraction of decode time spent in FFN gate exchange;
- \(F_p\) be the measured fraction of prefill time spent in FFN big-gate exchange;
- \(W\) be the fraction of that gate time that scales with payload bytes;
- \(C\) be new pack/unpack time as a fraction of the original total workload time.

Then the plausible total savings are approximately:

\[
S_d \le 0.5 \times F_d \times W - C_d
\]

\[
S_p \le 0.5 \times F_p \times W - C_p
\]

The hard ceiling is \(0.5F\), achieved only when the entire gate cost scales with bytes and conversion is free. No checked profiling result provides trustworthy current values for \(F_d\) or \(F_p\), so a numeric whole-inference percentage cannot be honestly asserted from this source alone.

A useful reporting format after implementation would be:

| Workload | Baseline total | Baseline FFN-gate total | Payload-scaled portion | Pack | Unpack | Candidate total | Net saving |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decode, one token | measured | measured | measured | measured | measured | measured | measured |
| Prefill, short | measured | measured | measured | measured | measured | measured | measured |
| Prefill, maximum chunk | measured | measured | measured | measured | measured | measured | measured |

### Reasons the realized saving may be smaller

- Pack and two symmetric unpack operations add GPU memory traffic and kernel launches.
- The service thread still posts work requests, polls completions, checks peer liveness, and releases the gate.
- Decode’s fixed per-gate synchronization occurs 75 times per token, independent of payload size.
- RDMA provider and kernel-driver overhead may be latency-dominated rather than byte-dominated.
- The canonical F32 add remains unless later fused with unpack.
- Direct packed-buffer visibility to OdinLink may fail, forcing another staging step.
- If F16 is rejected by range or accuracy tests, BF16’s coarser mantissa may fail the proposed tolerance despite identical performance potential.
- Prefill can be dominated by routed-expert computation, attention, or non-wire portions of the big gate.
- The historical measurement says non-direct staging consumed roughly 64–65% of big-gate time at about 200 MB/s (`ds4_rocm.cu:622-633`, `ds4_tp.c:582-596`). Halving the bytes read by that path could at most halve that component, suggesting a prefill-gate saving near 32% of the old gate time before conversion overhead—not 50%—if the same profile still applies. Since the BIG_DIRECT work may have changed which portion dominates, this figure is historical context rather than a current forecast.
- Compression may shift the bottleneck from memory movement to completion latency, making the second half of the byte reduction worth less than the first.

The honest expectation is therefore: potentially material for prefill if the measured gate remains bandwidth/staging dominated; plausibly modest for decode unless converting two 16 KiB messages into one 12 KiB message removes meaningful fixed overhead; and unknown at whole-engine level until `F_d`, `F_p`, pack cost, and unpack cost are measured. The lever should proceed only after agreement on the section 3 accuracy tolerance, and it should ship—if it passes—as an explicitly approximate, opt-in mode.

Codex session ID: 019fd452-c2de-7453-973f-339455a04b2d
Resume in Codex: codex resume 019fd452-c2de-7453-973f-339455a04b2d
