# Where the roofline numbers come from

Written because the numbers were questioned, which they should have been - the
first two versions of this analysis were both wrong.

## Every input, and its source

| quantity | value | source | status |
|---|---|---|---|
| routed experts (all) | 145.12 GiB | parsed from our GGUF header, summed by tensor dims x quant block size | computed |
| attention + indexer + compressor | 5.49 GiB | same | computed |
| other dense (shexp, norms, hc) | 1.20 GiB | same | computed |
| output projection `output.weight` | 0.524 GiB | same | computed |
| `token_embd.weight` | 0.986 GiB | same - EXCLUDED, it is a row lookup not a stream | computed |
| experts active per token | 6 of 256 | model config | given |
| **active bytes per token** | **10.62 GiB** | 145.12x6/256 + 5.49 + 1.20 + 0.524 | derived |
| **achievable read bandwidth** | **223.9 GiB/s** | **MEASURED**, scripts/t6_bandwidth_probe.cpp | measured |
| decode throughput | 6.5 t/s | measured across runs (6.18, 6.56, 6.92) | measured |

Validation that the byte accounting is right: the buckets sum to 153.32 GiB
against a 153.33 GiB file - a 0.005 GiB gap that is metadata and padding.

## Roofline

| config | bytes/node | ceiling |
|---|---|---|
| no TP, one node | 10.62 GiB | 21.1 t/s |
| **experts sharded only (today)** | 8.91 GiB | **25.1 t/s** |
| experts + attention split | 6.17 GiB | 36.3 t/s |

**Measured 6.5 t/s = 26% of our own ceiling.** llama.cpp does 15 t/s here.

## Three corrections made while establishing this

1. **Wrong checkpoint.** The first version used active-byte figures measured for
   UD-Q8_K_XL (161.9 GB) and applied them to our Q4_K file. That inflated
   attention from 5.49 to 9.45 GiB and inverted the conclusion.

2. **Missing the output projection.** `token_embd` and `output` were bucketed
   together and both excluded. `token_embd` is correctly excluded (one row read
   per token); `output.weight` is NOT - the logit projection reads all 0.524 GiB
   every token. +5% to active bytes.

3. **The bandwidth was never measured.** 195.6 GiB/s came from converting a
   "~210 GB/s effective" note inherited from earlier work. Measured today it is
   **223.9 GiB/s (240.5 GB/s), 94% of the 256 GB/s theoretical peak** - so the
   inherited figure was ~7% conservative.
   The first version of the probe reported **996 TB/s** because the accumulator
   was only consumed under `threadIdx.x == 0xffffffffu`, provably false for a
   256-thread block, so the compiler deleted every load. The sink is now
   compared against a value supplied by the host at runtime.

## Caveats that remain

- A roofline assumes decode is purely bandwidth-bound with perfect overlap. It
  is an UPPER BOUND, not a prediction. Dependency stalls, kernel launch gaps and
  the 86 synchronous gates per token are all excluded by construction - which is
  precisely why the 26% gap is the interesting number.
- The 223.9 GiB/s is a pure sequential streaming read. Real weight access is
  strided/quantised and will achieve less.
- Per-node figures assume the expert shard halves the expert bytes exactly,
  which the 80.76 GiB residency measurement supports.
