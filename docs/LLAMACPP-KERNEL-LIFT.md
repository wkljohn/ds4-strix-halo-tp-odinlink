# Can we lift llama.cpp's Q4_K matrix-core kernel? Three preconditions, all met.

Checked directly against local llama.cpp checkouts
(`models/llama.cpp-hyv3` @ 1a064ab, 2026-07-22) and the ds4 tree.

## 1. Block layout: BYTE-IDENTICAL

| | llama.cpp `block_q4_K` | ds4 `cuda_block_q4_K` |
|---|---|---|
| | `ggml_half d` | `uint16_t d` |
| | `ggml_half dmin` | `uint16_t dmin` |
| | `uint8_t scales[K_SCALE_SIZE=12]` | `uint8_t scales[12]` |
| | `uint8_t qs[QK_K/2=128]` | `uint8_t qs[CUDA_QK_K/2=128]` |
| **size** | **144 B** | **144 B** |

`QK_K == CUDA_QK_K == 256`, `K_SCALE_SIZE == 12`. Both consume the same GGUF
Q4_K, and ds4's own code already assumes 144 B/block elsewhere.

**A lifted kernel can read our resident weights directly - no conversion, no
re-quantisation, no extra memory.**

## 2. gfx1151 IS on llama.cpp's matrix-core path

`ggml/src/ggml-cuda/common.cuh`:

    static bool amd_wmma_available(const int cc) {
        return (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3(cc));
    }

and RDNA3.5 is explicitly recognised:

    #define GGML_CUDA_CC_RDNA3_5 (GGML_CUDA_CC_OFFSET_AMD + 0x1150) // AI 370, AI Max 395 laptops.

That comment names our exact part. llama.cpp routes Q4_K through MMQ
(`mmq.cuh:81, 385, 576, 734`) and gets WMMA on this silicon; ds4 uses DP4A.

## 3. Licence: MIT->MIT, and ds4 has ALREADY done this

- llama.cpp: `MIT License, Copyright (c) 2023-2026 The ggml authors`
- ds4 `LICENSE`: `MIT License / Copyright (c) 2026 The ds4.c authors /
  **Copyright (c) 2023-2026 The ggml authors**`

ds4's README states it outright (README.md:53-56):

  "Some source-level pieces are retained or adapted here under the MIT license:
   GGUF quant layouts and tables, CPU quant/dot logic, and certain kernels. For
   this reason ... we keep the GGML authors copyright notice in our LICENSE file."

**Porting a ggml kernel into ds4 is precedent, not a novel legal question**, and
the required attribution is already present. Any new port should still be
commented with its provenance.

## The one serious counter-indication

llama.cpp issue **#17917** reports a **~40% pp regression** (pp2048 900 -> 543)
*after* WMMA-MMQ was enabled for RDNA3, and **#21284** (gfx1151 prefill defaults)
was **closed unmerged**. So llama.cpp's own RDNA3 WMMA path is contested and may
not be the thing delivering its 80-95 t/s here.

This matters more than the three green lights above: if their WMMA path is a
regression on RDNA3, lifting it could reproduce the regression rather than the
speedup. **Establish which path llama.cpp actually takes for Q4_K on gfx1151 at
prefill batch sizes - MMQ-WMMA, MMQ-DP4A, or dequant+hipBLAS - before porting
anything.** That is a measurement on the local checkouts, not a code read.

## Status

Preconditions met; the value is unproven. Two kernel predictions have already
failed in the wrong direction in this project, so the next step is measurement
(which path does llama.cpp use, and what does it achieve), not a port.

## RESOLVED (2026-08-05): llama.cpp does take MMQ-WMMA for this workload class

Codex source-level research (not a benchmark - a direct read of
`llama.cpp-upstream-latest`'s dispatch code) confirms the open question
above: for a 256-expert MoE like ours, llama.cpp's own routing rule
(`ggml-cuda/mmq.cu:347`) forces custom MMQ whenever `n_experts >= 64`
("per-expert BLAS synchronization is expensive" at that count), and Q4_K
defaults to MMQ regardless. Combined with this doc's own earlier finding
(`amd_wmma_available()` covering RDNA3/RDNA4, and the RDNA3.5-specific Q4_K
tile config table at `ggml-cuda/mmq-config-rdna3-5.cuh:116`), MMQ on
gfx1151 is WMMA, not DP4A, for exactly the model class this project cares
about. **The counter-indication above (llama.cpp issue #17917's RDNA3
WMMA regression) does not appear to override this for MoE-shaped
workloads** - the regression report was for a different scenario and
llama.cpp's own dispatch code still routes n_experts>=64 through MMQ on
this hardware regardless.

**Practical consequence, already realized independently**: this project's
own down-projection WMMA fix (`ds4-upstream@8b71a30`, same session) means
"llama.cpp uses matrix cores while ds4 uses only DP4A" is now OBSOLETE for
Q4_K - ds4's routed Q4_K gate/up/down all use integer WMMA with a Q4_K
weight + Q8_1-style activation design, the same basic family as llama.cpp's
MMQ-WMMA. The remaining, now much narrower gap is NOT "matrix cores vs
none" but specific dispatch-breadth differences:

- ds4's WMMA tile is fixed at `J=16` and requires >=6 routed pairs/expert
  to engage (cold experts fall back to DP4A); llama.cpp's `mul_mat_q_switch_J`
  dynamically searches widths from 8 through 128 and picks whichever needs
  the fewest column tiles, plus stream-K load balancing
  (`ggml-cuda/mmq.cuh:1388,1469`).
- On a SHORT prompt (~50 tokens, this project's usual quick test), 6
  selected experts/token across 256 total averages only ~1.17 routed
  pairs/expert - most experts can't even reach ds4's 6-pair WMMA
  threshold, so most of a short prefill runs on the DP4A cold path
  regardless of the WMMA port's existence. This project's own earlier
  analysis (`Q4K-WMMA-PLAN.md:7`) already anticipated this scaling: ~22
  and ~48 pairs/expert at 945 and 2048 tokens respectively, with
  llama.cpp's advantage EXPECTED to grow with batch size (i.e. the gap
  should look different, not necessarily worse, at a realistic long-prompt
  comparison than at this project's usual short smoke-test prompt).

See `WHY-VLLM-PREFILL-IS-6X.md` for the original "29.5 vs 80-95 vs 198.8"
comparison table this resolves context for. **The 80-95 t/s llama.cpp
figure is confirmed golden evidence, witnessed firsthand at the node by
this project's operator** - not an unprovenanced claim (an automated
research pass raised that flag; it is withdrawn, see
`WHY-VLLM-PREFILL-IS-6X.md`). What remains open is a fair, matched
re-measurement of ds4's CURRENT (post-WMMA-fix) prefill at a comparable
prompt length (pp512 or longer, repeated samples) against that same
llama.cpp figure - the exact multiple may differ from the original "29.5
vs 80-95" once ds4's own WMMA fix and a longer, less-fragmented-routing
prompt are both accounted for, but the llama.cpp reference point itself is
not in question.
