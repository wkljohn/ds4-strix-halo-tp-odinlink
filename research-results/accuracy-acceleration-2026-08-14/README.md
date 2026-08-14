# Accuracy acceleration A/B artifacts

This directory contains the raw evidence for
`docs/ACCURACY-ACCELERATION-IMPACT-2026-08-14.md`.

The principal files are:

- `q2-current-full.tsv` and `q2-iq2-disabled-full.tsv`: causal IQ2 A/B;
- `q2-both-wmma-disabled-full.tsv`: attribution control;
- `q4-current-full.tsv` and `q4-wmma-disabled-full.tsv`: short-prompt Q4 A/B;
- `q4-frontier-*/frontier_002048.logits.json`: long-prefill full logits;
- `coordinator-*.log` and `worker-*.log`: two-rank RDMA evidence;
- `remote-*-request.json` and `remote-*-response.json`: secondary llama.cpp
  semantic probes.

All complete TP runs used explicit OdinLink RDMA and recorded
`fallback_calls=0`. The aborted `q2-current-pilot` log is retained because it
exposed the scorer teardown ownership bug; `q2-current-pilot2` is the clean
post-fix reproduction.
