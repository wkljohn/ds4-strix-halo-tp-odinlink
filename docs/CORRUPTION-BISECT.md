# Corruption hunt: what is ruled OUT (and how)

Deterministic reproducer (greedy, so sampling is not a variable):

    --temp 0 -p "What is the capital of France? Answer with one word."
    -> "We核心Apacser##- 1 H Linesit<originalden> 1804-<｜f- ## * Ir浦er ..."
    prefill ~30 t/s, generation ~6.5 t/s

The FIRST token is already wrong, so the fault is in prefill, not decode.

## Ruled out

1. **Sampling.** Earlier runs differed from each other; that was temperature 1.0
   (the CLI default), NOT a race. Under `--temp 0` the output is byte-identical
   across runs. I should have set this before ever calling the output corrupt.

2. **Patch 15 double-counting the addend.** I was about to "fix" this and it
   would have BROKEN a correct path. `ds4.c:23905` passes
   `metal_graph_shared_out(g)` as the addend and the comment states the intent:
   "local partial = shared expert + owned routed experts". The shared expert is
   itself K-split (`tp_split_shared`), so each rank's `shared_out` is only ITS
   half - summing the two partials reconstructs the whole. Folding it on both
   ranks is correct by design.
   Verified the whole chain is coherent: gate/up write this rank's lanes
   "compact at the buffer base" (ds4.c:23695), so the down K-slice's
   `x_elem_off = 0` with `k_off = rank*(shared_dim/2)` is right.

3. **The attention head split.** `DS4_GLM_TP_HEAD_SPLIT_MIN` exists precisely
   "for correctness isolation". Forcing it to 99999999 makes attention replicate
   instead of split. Output was **byte-identical**, so the head split was not
   engaged for this model and cannot be the cause.
   (It IS a latent bug for models that do use it: `g_tp_attn_head_split` is
   written at ds4_rocm.cu:476 and never read, while the one-per-chunk zeroing at
   ds4.c:44002 assumes the kernels only write owned heads. Recorded, not fixed.)

4. **Expert sharding being inactive.** `g_tp_split_world = 2` is set in
   `ds4_gpu_tp_init` (ds4_rocm.cu:342), which demonstrably runs.

## Method correction

Four hypotheses, four refutations. The cheap ones were the ones that used an
existing switch or an existing probe; the expensive ones were the ones I argued
from reading code. The head-split test cost one run and no code change and
killed a theory I would otherwise have spent hours implementing.

**Stop hypothesising. Get a reference.**

## Next: an actual reference run

There has never been a known-good baseline for this model on this backend, so
every "is TP wrong?" question is unanswerable. `--ssd-streaming` streams weights
from SSD instead of requiring full residency, which should allow a SINGLE-NODE,
NO-TP run of the same 153 GiB checkpoint on one 96 GiB node.

That would separate two possibilities that no amount of code reading can:
- single node produces "Paris" -> the ROCm backend is sound and TP is at fault
- single node is ALSO garbage -> the fault is in the ROCm backend (or the
  checkpoint/flags), and TP is a red herring

Note `tp_shard` requires `!ssd_streaming`, so these are mutually exclusive by
construction - which is exactly why this is a clean control.
Also re-test without `DS4_CUDA_NO_Q8_F16_CACHE=1`, which I introduced as a
memory workaround and have never validated for numerics.

## Update: the --ssd-streaming control does NOT exist

Tried it. Fails at layer 3 (the first MoE layer; 0-2 are dense):

    ds4: ROCm SSD streaming routed MoE missing compact selected experts
         (layer=3 tokens=1 total_experts=256 selected=6); full expert table is not mapped
    ds4: ROCm routed_moe_ONE failed: layer=3 shard=0 n_total=256 n_exp=6
         gate_type=12 down_type=12 in=4096 mid=2048 out=4096

SSD streaming and the generic Q4_K routed-MoE path are mutually incompatible on
ROCm: streaming requires a compact selected-expert mapping that this path does
not provide. Note `shard=0` - no TP involved - so this is an upstream ROCm
limitation, not one of my patches.

Consequence: there is NO single-node reference run of this checkpoint on this
hardware. It does not fit resident (153 GiB vs 96 GiB) and it cannot stream.
That is precisely why TP exists here, and it means "compare against a
known-good ds4 run" is unavailable as a technique.

## The reference that DOES exist: llama.cpp

llama.cpp runs this exact Q4_K checkpoint on this exact hardware at 15 t/s
(and 16.6 t/s with DSpark two-node). It is ground truth for:
- the expected greedy continuation of a given prompt, and
- the throughput bar.

Use it for output comparison instead of chasing a ds4 baseline that cannot be
built. It does NOT localise the fault to a layer, so it complements rather than
replaces per-stage instrumentation.
