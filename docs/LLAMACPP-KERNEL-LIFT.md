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
