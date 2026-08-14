# Q2 IQ2_XXS integer-WMMA campaign

The tracked design and accepted results are documented in
`../../docs/Q2-I8-WMMA.md`. Raw benchmark artifacts are retained locally under
`../quant-comparison-2026-08-10/`.

Key accepted artifacts:

- `q2-iq2-i8-final-perf-r1-20260814.csv`
- `q2-iq2-i8-final-perf-r2-20260814.csv`
- `q2-iq2-i8-final-perf-r3-20260814.csv`
- `coordinator-premain-q2-20260814T095553Z.log`
- `worker-premain-q2-20260814T095553Z.log`
- `premain-q2-20260814T095553Z.csv`
- `coordinator-premain-q4-20260814T095553Z.log`
- `worker-premain-q4-20260814T095553Z.log`
- `premain-q4-20260814T095553Z.csv`

Final Q2 median: 162.78 prefill t/s and 14.68 decode t/s on the fixed
2,048+300 workload, fingerprint `f9cb3a8a17e95c71`.
