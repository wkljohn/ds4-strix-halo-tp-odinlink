# Warm-run performance diagnosis

Repeated prompts did **not** progressively slow the validated ROCm 7.14 build.
Three new-process runs of the identical 2,048+300 workload measured
305.02/21.12, 333.22/21.18, and 334.75/21.17 prefill/decode t/s. All used the
same binary, prompt, model, RoCE v2 profile, and fingerprint
`0163c44015591445`.

A second test sent ten equal-size, independent 2,842+64 requests through one
long-lived `ds4-server` session. Wall time stayed between 13.19 and 13.74 s;
after the first warm request, prefill stayed near 266--268 t/s and decode near
20.98--21.03 t/s. The live slot correctly reported a prefix miss and rebuilt
each independent prompt, so prefix reuse did not conceal a decline.

The earlier 332.52/21.16 to 297.60/19.27 same-process change was a 2,048-row
versus 4,096-row context comparison. It measures the expected cost of a longer
attention/KV history, not cumulative machine degradation. The separate
301.31/19.57 result was built with ROCm 7.2 and is excluded from the ROCm 7.14
repeat series.

The full evidence, Fable review, ranked hypotheses, and confirmation matrix
are kept at the canonical research location:

`$DS4_RESEARCH_ROOT/diagnostics/warm-run-degradation-20260905/REPORT.md`

Fable 5.1 and Grok 4.6 independently reviewed the evidence and agreed that no
allocator, kernel, clock-policy, or RDMA source change is justified. The local
benchmark configuration now pins the validated ROCm 7.14 ELF toolchain
fingerprint so a stale 7.2 binary fails closed. A source fix should be attempted
only if equal-work fixed-context requests reproduce a decline and telemetry
identifies thermal/DVFS, UMA/GTT fragmentation, session lifetime, or RoCE
backpressure as its owner.
