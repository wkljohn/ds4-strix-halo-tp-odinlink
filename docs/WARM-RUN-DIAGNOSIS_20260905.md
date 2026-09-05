# Warm-run performance diagnosis

The current evidence does not establish that repeated prompts slow DS4 until
reboot. A same-process frontier sweep fell from 332.52/21.16 t/s at 2,048
rows to 297.60/19.27 at 4,096 rows, which is a larger-context comparison.
The fresh fixed 2,048+300 run measured 301.31/19.57 t/s. The direct-flash
follow-up was aborted before a timing row.

The full evidence, Fable review, ranked hypotheses, and confirmation matrix
are kept at the canonical research location:

`$DS4_RESEARCH_ROOT/diagnostics/warm-run-degradation-20260905/REPORT.md`

No source fix is promoted until identical fixed-frontier runs reproduce a
within-process or cross-process decline and identify whether the cause is
thermal/DVFS, UMA/GTT fragmentation, session lifetime, or RoCE backpressure.
