# The Q4_K MoE prefill kernel, from the compiled gfx1151 object

Independently disassembled (`llvm-objdump --offloading` then
`-d --mcpu=gfx1151`) rather than reasoned about. Reproduced by me after a
reviewer reported the same numbers; every count matches exactly.

## `moe_gate_up_mid_q4K_expert_tile8_row32_kernel` (rocm/ds4_rocm_moe.cuh:1754)

| opcode | count | per dot4 |
|---|---|---|
| **total body instructions** | **2686** | — |
| `v_dot4_i32_iu8` (the actual work) | **128** | 1.0 |
| `v_movrels_b32` / `v_movreld_b32` | **116** | 0.91 |
| `v_and_b32` + `v_lshrrev_b32` | **195** | 1.52 |
| `s_delay_alu` + `s_nop` | **386** | 3.0 |
| `flat_load*` | 69 | — |
| `ds_store*` | 37 | — |
| **`ds_read` / `ds_load`** | **0** | — |
| `scratch_store*` | 30 | — |

**128 of 2686 instructions - 4.8% - do useful arithmetic.**

Three defects, each visible in the ISA:

1. **116 `v_movrel`**: `dev_dot_q4_K_q8_K_block8` (moe.cuh:321) takes the tile
   width as a RUNTIME argument `uint32_t n` and loops `for (p=0;p<n;p++)` with
   `if (!ys[p]) continue;` (:344-349). The compiler cannot bound `n <= 8`, so
   `isum[8]`/`summs[8]`/`ys[8]` become dynamically indexed - M0-serialised
   indexed register moves that block dual-issue, nearly one per dot4.
2. **195 and/lshr**: the Q4_K nibble unpack (`dev_dot_q4_32`, :264) sits INSIDE
   the p loop, so the same 8 weight bytes are unpacked once per token instead of
   once per tile.
3. **0 `ds_read` despite 37376 B of LDS**: `xqb[p]` is assigned either a global
   or a `__shared__` pointer depending on `xq_blocks <= 16u` (:1798-1806), so it
   is a *generic* pointer and every activation read becomes a `flat_load` with
   runtime address-space resolution. The kernel pays 37 KB of LDS - which caps
   occupancy at 37.5% in WGP mode (128 KB/WGP / 37376 = 3 workgroups = 24 of 64
   wave32) - and reads none of it.

## How this squares with the tile A/B

Measured: tile_m=4 gives 27.10 t/s, tile_m=8 gives 30.00 t/s. Narrowing hurts.
That looked like it contradicted "instruction-bound", but it does not - the two
act on different terms:

- **Tile width sets the ceiling.** 8 tokens per weight stream => arithmetic
  intensity 26.4 flop/byte => ~5.9 TFLOP/s for that shape. tile4 halves the
  tokens per stream, roughly halving the ceiling; the occupancy gain (37.5% ->
  75%) does not make that back.
- **Instruction overhead sets how much of the ceiling is reached.** 925 GFLOP/s
  is **16% of the tile8 ceiling**, and 4.8% useful instructions explains it.

So both fixes are real and independent: fix the instruction mix to approach the
current ceiling (up to ~6x headroom), and widen the tile with K-tiling to raise
the ceiling. Neither is an occupancy problem.

## Peak, with the right denominator

`v_dot4_i32_iu8` is a full-rate VALU op: 64 lanes/CU/clk x 4 MAC x 2 flop x
40 CU @ 2.9 GHz = **59.4 TOPS**. Against that, 925 GFLOP/s is **1.56%**.
(An earlier note in this repo compared against fp32 peak, 14.8 TFLOP/s, and
reported 6% - wrong denominator, too generous.)

## Patch 23 (templating the p loop) — IMPLEMENTED, MEASURED, REVERTED

The recommended first fix was to make the pair count a template parameter so the
`p` loop unrolls and the dynamic indexing disappears. Verified bit-identical by
construction (all callers declare `xqb[8]` all-NULL, fill `[0,np)`, and both
loops skip NULL, so a compile-time 8 visits the same slots in the same order).

The ISA moved exactly as predicted:

| metric | before | after |
|---|---|---|
| total body | 2686 | 7680 |
| `v_dot4` (real work) | 128 | **1024** |
| `v_movrel*` | 116 | **16** (-86%) |
| `v_and` + `v_lshrrev` | 195 | **1541** |
| `scratch_store*` | 30 | **0** |
| useful-instruction share | 4.8% | **13.3%** |

**And prefill got SLOWER: 27.91 t/s vs 29.91 baseline (-7%).** Output stayed
coherent, so the change is correct - just worse.

Why: unrolling traded 116 `v_movrel` for **1346 extra and/lshr**. The Q4_K
nibble unpack (`dev_dot_q4_32`, moe.cuh:264) sits INSIDE the `p` loop, so
unrolling replicated it 8x. Total ALU work rose even though the *ratio* of
useful instructions improved, and the body grew 2.9x (I-cache pressure).

**Lesson: the unpack hoist is a PREREQUISITE for templating, not a follow-up.**
The estimate of "+2.5x for half a day, low risk" was wrong because it treated
the two as independent and separable. They are not: unrolling without hoisting
is strictly negative.

Reverted. Any retry must hoist the weight unpack out of the `p` loop in the same
change - load and unpack `x->qs + byte_off` ONCE per `j`, then dot it against
each of the N activations - so the unpack count stays ~193 while `v_dot4`
reaches 1024. That is a real restructuring of `dev_dot_q4_K_q8_K_block8`, not a
signature change, and it should be verified bit-identical against the greedy
reproducers before being trusted.
