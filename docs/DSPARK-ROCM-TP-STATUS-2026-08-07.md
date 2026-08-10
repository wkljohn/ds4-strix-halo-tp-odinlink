# DSpark on ROCm TP=2: Architecture and Current Status

Date: 2026-08-07

This report describes DSpark's role in inference, the DeepSeek reference and
llama.cpp implementations, the original DS4 path, and the current two-node
ROCm/OdinLink implementation. The benchmark sweep was paused at the user's
request; the four-token cap point did not finish and is not reported.

## Executive summary

```text
Normal decode
target token -> target token -> target token -> target token
   serial          serial          serial

DSpark decode
target anchor -> draft several cheap candidates -> verify them together
                                              -> accept exact matching prefix
```

DSpark is a speculative decode accelerator, not a prompt-prefill accelerator.
It is functionally correct in this fork and often predicts the target well,
but it is not yet faster than ordinary target-only decode on the two-node
gfx1151 setup. The latest short diagnostic reached 4.66 generation t/s versus
11.64 t/s target-only. The primary remaining problem is architectural: the TP
verifier currently turns each speculative row into a separate single-token
routed-MoE call inside every target layer. The existing ROCm batch MoE path
already supports TP expert sharding, so the next experiment should reuse it
behind an opt-in A/B switch rather than create a new kernel.

## 1. What DSpark does

DSpark is a lossless speculative-decoding sidecar. It does not replace the
target model and does not accelerate prompt prefill. It tries to replace
several serial target decode steps with one draft pass and one parallel target
verification pass.

```text
prompt / committed tokens
          |
          v
 full target decodes one anchor token
 and exposes selected hidden states
          |
          v
 DSpark parallel backbone
 (three support stages, five positions)
          |
          v
 five base-logit rows
          |
          v
 low-rank Markov chain
 logit[i] += W2 * W1[previous token]
          |
          v
 candidate block: d1 d2 d3 d4 d5
          |
          v
 full target verifies candidates as a batch
          |
          v
 accept the exact matching prefix; discard the rest
```

The expensive support backbone is parallel across draft positions. The small
Markov head is sequential across those positions and restores dependence on
the preceding proposed token. The target remains authoritative, so greedy
output is identical to target-only greedy decode when state handling is
correct.

The official design also predicts a confidence score for each position and
uses a hardware/load-aware scheduler to avoid verifying low-yield suffixes.
That is primarily a serving-throughput mechanism. This fork deliberately keeps
the main DSpark confidence threshold disabled and performs fixed-length, exact
verification. Unrelated internal confidence heuristics are unchanged.

## 2. Reference implementations

### DeepSeek reference

DeepSeek's reference V4 DSpark model:

1. Captures selected target hidden states and fuses them with `main_proj` and
   `main_norm`.
2. Builds a noise-token block anchored by the latest committed token.
3. Runs the block through the DSpark stages in parallel.
4. Collapses the final hyper-connection state and applies the shared target
   output head to produce one base-logit row per draft position.
5. Applies the low-rank Markov head left-to-right and predicts confidence from
   the draft hidden state plus the Markov embedding.
6. Sends a selected prefix to the target verifier.

Primary sources:

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/inference/model.py
- https://arxiv.org/abs/2607.05147
- https://github.com/deepseek-ai/DeepSpec

### Current llama.cpp

Current llama.cpp implements DSpark as an extension of its DFlash speculative
path. It uses separate target and draft contexts:

1. The target context exposes the configured target-layer features.
2. The DFlash encoder fuses those features and injects them into the draft KV
   cache.
3. One draft `llama_decode()` evaluates the complete anchor/noise block.
4. The graph applies Markov bias, argmax chaining, and confidence on-device.
5. llama.cpp's central speculative decoder performs the existing target
   verification and prefix-accept operation.

The initial merged work reused the DFlash graph and verifier. Current source at
commit `cb26014d965036f0eacdab7387b222d28d36b9e6` also contains a DeepSeek-V4
specific `graph_dsv4` with full V4 hyper-connection, MLA, and MoE stages. The
checked documentation still says Qwen3-only, so source is ahead of that page.

Primary source:

- https://github.com/ggml-org/llama.cpp/pull/25173
- https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/models/dflash.cpp

### Original upstream DS4

The inspected upstream DS4 point is `b030961` (2026-08-05). Its V4 path already
has the essential DSpark algorithm:

- target-hidden capture from layers 40, 41, and 42;
- three DSpark stages and a trained five-token block;
- support KV continuity and non-causal block attention;
- Markov correction and confidence-prefix selection;
- batched target verification and exact longest-prefix acceptance.

Its intended resident/multi-tier setup installs the support tensors in GPU
memory where possible. Its default user confidence threshold is 0.7. At that
point, accepted verifier rows are rolled back and replayed through ordinary
single-token target decode to make compressor/indexer state identical to the
normal path. That is safe but duplicates target work.

## 3. Current ROCm + cross-node TP implementation

```text
Node 0 / coordinator                         Node 1 / worker
--------------------                         ---------------
50% of target experts                       50% of target experts
target hidden capture
three-stage DSpark proposal (local)
compact selected support experts from NVMe
GPU Markov chain
             |
             +---- draft token block -----> mirrored verify command
             |
target batch half <====== OdinLink ======> target batch half
             |
coordinator compares target tops with drafts
             |
commit full/prefix state ---- command ----> commit matching worker state
```

Important differences from the resident single-system implementations:

| Area | Original DS4 / llama.cpp style | Current ROCm TP=2 fork |
| --- | --- | --- |
| Target weights | Resident or normal device placement | Each node holds a 50/50 routed-expert shard; measured 80.76 GiB of 153.32 GiB per rank |
| Filesystem | One local model view | Filesystems are not shared; identical model/support files and binaries are required on both nodes |
| Draft backbone | Normally resident on one or more local GPUs | Runs only on coordinator; a distributed draft-backbone experiment was slower and removed |
| Q8 support size | Original DS4 documentation describes a smaller support package | Current Q8 support GGUF is 10.15 GiB and cannot be kept as another full resident copy within the safe memory margin |
| Expert handling | Resident support tensors | Route first, deduplicate selected experts, then read only their gate/up/down slices for each of three stages |
| Storage path | Normal mapped/resident access | 16 bounded readers, requests sorted by file offset, buffered reads followed by page eviction; at most about 68 MiB pinned staging and no expert-weight cache |
| Markov head | Older DS4 fixed-block path read full logit rows and evaluated much of the chain on CPU | Base logits stay on GPU; Q8/BF16 projection, bias add, and argmax run on GPU with a correctness fallback |
| Target verification | One target batch | Same logical batch is mirrored on both TP ranks; every target layer combines the two expert halves over OdinLink |
| Accepted state | Upstream DS4 rolls back and replays accepted rows | Full accepts commit directly; the common `N-1` accepted prefix commits a captured verifier frontier directly; fallback replay remains for other partial shapes |
| Confidence | DS4 default 0.7; llama.cpp default threshold 0 | Main DSpark confidence threshold forced to 0 as requested; exact fixed-block verification |

The worker does not need the support backbone to produce proposals. It does
need the support GGUF locally for matching startup/model state, and it must run
the target verifier in lockstep so target KV, compressed attention,
compressor, and indexer state stay synchronized.

## 4. What has improved

All values below use the short, deterministic `Hi.` / 10-token diagnostic.
They characterize decode control flow, not long-prompt prefill throughput.

| Checkpoint | Draft proposal | Verify | Replay | Acceptance | Generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier correct local-backbone control | 4558.64 ms | 1758.76 ms | 282.41 ms | 8/9 | 1.33 t/s |
| Current five-token path | 1016.16 ms | 1747.43 ms | 0.00 ms | 8/9 | 3.05 t/s |
| Fixed cap 2 diagnostic | 1516.47 ms / 3 cycles | 839.57 ms / 3 verifies | 0.00 ms | 6/6 | 3.39 t/s |
| Fixed cap 3 diagnostic | 1255.84 ms / 3 cycles | 634.23 ms / 2 verifies | 0.00 ms | 6/7 | 4.46 t/s |
| Target-only control | n/a | n/a | n/a | n/a | 11.64 t/s |

The cap-3 result is a promising diagnostic, not a release result: one of its
three cycles missed the first draft and therefore skipped verification. A
longer matched test is required. The cap-4 run was stopped during startup for
this report.

Measured improvements:

- GPU Markov work fell from about 2076 ms to about 78-82 ms for the two-cycle
  five-token test.
- Sorted bounded expert reads removed the multi-second cold stage outlier.
- The latest six support-stage copies were 51-143 ms each while transferring
  127.5-267.75 MiB of selected weights.
- Direct full-block and common four-of-five prefix commits removed replay from
  roughly 282-1347 ms to zero in the tested path.
- Output remained exactly `Hello! How can I assist you today?`, with no
  verifier errors and 8/9 accepted draft tokens.

## 5. Current bottlenecks

For the current five-token diagnostic, two speculative cycles produced eight
accepted draft tokens:

```text
proposal total       1016 ms  #######################
  support chain       875 ms  ####################
  Markov               82 ms  ##

target verification  1747 ms  ########################################
  verifier layers    1747 ms  ########################################
  upload/read/head     <1 ms  .

rollback/replay          0 ms
ordinary anchor work   157 ms  ####
```

### 1. Target verification is the largest measured cost

The latest instrumented five-row verifier measured:

| Component across 43 target layers | Time | Share |
| --- | ---: | ---: |
| Total target-layer loop | 342.98 ms | 100% |
| OdinLink TP batch-gate exchange | 48.54 ms | 14.2% |
| Local target-layer work around the gate | 294.44 ms | 85.8% |

The worker showed the same shape: 340.57 ms in the layers and 43.67 ms in TP
gates. The network is therefore not the dominant verifier cost.

Source inspection found a concrete cause. In the cross-node verifier branch,
each target layer loops over all speculative rows and calls the one-token
routed-MoE path once per row. A five-row block therefore performs five
single-token MoE calls per layer before one batched TP exchange. This defeats
the batch amortization that speculative verification is supposed to provide.

The ordinary ROCm batch path already calls `ds4_gpu_routed_moe_batch_tensor()`,
and that function already remaps every selected expert pair to the local TP
shard. This makes an opt-in switch to the existing batch path the best next
candidate. It needs correctness and performance A/B testing; it has not yet
been enabled or claimed as an improvement.

### 2. The draft backbone is still too expensive even after I/O repair

The proposal averages about 508 ms per cycle in the five-token run. About 437
ms per cycle is the three-stage support chain. Selected expert storage traffic
is now bounded and stable, but it is still real traffic on every speculative
cycle because the full 10.15 GiB Q8 support model cannot safely remain resident.

### 3. Single-stream economics are currently negative

The target-only control takes about 86 ms per generated token in this tiny
test. DSpark accepts four draft tokens per cycle, but currently spends roughly
508 ms drafting plus 874 ms verifying each cycle. High acceptance therefore
does not translate into speedup: the work used to earn four tokens costs much
more than four ordinary target decode steps.

### 4. Running the support backbone across both nodes did not help

The experimental TP support backbone increased the two-cycle support-chain
time from 2419 ms to 5738 ms in the matched early control. The draft has only
three stages, so its additional cross-node synchronization overwhelmed the
saved expert work. Keeping proposal generation on the coordinator is currently
the better architecture.

### 5. Variance remains high

Fresh runs warm roughly 80.76 GiB of each target shard and are sensitive to
file-cache, power, and residency state. Short output tests are useful for
correctness but insufficient for choosing a default. Any final setting needs
alternating, longer target-only/DSpark runs and VRAM telemetry.

## 6. Recommended continuation

1. Add an opt-in verifier switch that replaces the per-row one-token MoE loop
   with the existing TP-aware ROCm batch MoE function. Keep the current path as
   an immediate rollback.
2. Validate exact output and two-node state continuity at fixed verifier sizes
   2, 3, 4, and 5. Compare layer time, gate time, total verification time,
   acceptance, and replay count.
3. If batching is correct and faster, repeat the fixed-length sweep and choose
   the best block length using longer alternating target-only/DSpark runs.
4. Profile support-stage compute versus storage after the verifier is repaired;
   do not add a persistent support-weight cache.
5. Only consider a new kernel if the existing batch implementation is still
   dominant. Start any new kernel with isolated 2-row correctness, then
   3/4/5-row tests before inference integration.
6. Measure VRAM on both nodes and compare the final result against the
   11.64-13.8 t/s target-only range.

The current conclusion is: DSpark is functionally correct under ROCm TP=2, its
proposal acceptance is good, and several avoidable overheads have been
removed. It is not yet an acceleration on this hardware. The next decisive
work is to restore real batching in the TP target verifier, not confidence
tuning, not another support-weight cache, and not a new kernel unless the
existing batch path proves inadequate.
