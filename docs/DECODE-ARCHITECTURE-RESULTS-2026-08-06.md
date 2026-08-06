# Decode architecture results — 2026-08-06

This note records the retained measurements and rejected candidates from the
first post-prefill decode pass. Raw coordinator and worker logs are kept in the
local, Git-ignored `research-results/2026-08-06/` archive.

## Current reference

The promoted ROCm Q8 projection kernels raised the same TP=2 setup from the
original 9.96 decode t/s to 13.48 t/s in a 1,000-token stability run. A later
300-token transport-profile run measured 13.12 t/s; a same-prompt provider
control measured 13.76 t/s. Short-run variation is therefore material, and a
candidate is not promoted from a single run.

These are end-to-end `ds4` CLI measurements, not `ds4-bench` results.

## Q4_K routed-MoE ceiling

`DS4_ROCM_Q4K_DECODE_EVENT_PROFILE=1` enables a deferred 16-slot HIP event
pool. It does not synchronize the device on the launch path. The measured
per-layer means were:

| Stage | Rank 0 | Rank 1 |
|---|---:|---:|
| gate/up | 0.281 ms | 0.247 ms |
| middle quantization | 0.014 ms | 0.012 ms |
| down projection and sum | 0.113 ms | 0.106 ms |

The gate/up phase is already weight-bandwidth dominated: it streams about
56.6 MiB of distinct Q4_K weights per layer. Staging the shared Q8_K activation
in LDS regressed gate/up to 0.286/0.262 ms. A bounded launch-geometry check was
also rejected: 128 threads was neutral and rank-asymmetric (0.277/0.250 ms),
while 512 threads regressed to 0.299/0.271 ms. Both experiments preserved
generated output and were removed.

The 0.034 ms/layer gate/up difference also explains part of the asymmetric
FFN exchange wait: over 43 layers it is about 1.46 ms/token, roughly 2% of a
13.1 t/s token. A non-contiguous or replicated-expert ownership design is not
justified by this workload alone; it would need broader expert-frequency data
and a demonstrated >=5% end-to-end ceiling.

## OdinLink callback ceiling

With the service-interval profiler enabled, the typical slower-rank callback
means were 65.915 us for attention and 132.514 us for FFN after excluding one
recorded 1.0106-second scheduler/system outlier. Across 43 layers this is about
8.53 ms/token, or 11.2% of the observed 76.2 ms/token.

That callback bucket is not pure wire time. It includes rank arrival skew: the
rank that finishes its local expert work first waits inside the exchange. The
observed near-side transport floor is about 50–65 us/gate, so a provider-only
change must remove most of that floor to clear a 5% whole-token threshold.

## Rejected 16 KiB caller-thread send

OdinLink's provider normally sends payloads up to 4 KiB directly from the
calling thread. Raising that ceiling to the exact 16 KiB decode vector removed
decode gates from the provider's WC bounce-copy counters on both ranks, proving
that the intended path was active. It nevertheless regressed the same 300-token
workload from 13.76 to 11.75 t/s (-14.6%); prefill remained comparable at
158.86 versus 160.15 t/s. The worker queue supplies useful overlap with the
kernel stream-send ioctl, so this change was reverted.

## RDMA WRITE assessment

The current OdinLink provider accepts a verbs work request but its data plane
maps payload operations onto two-sided stream send/receive. The NHI receives
into kernel coherent frame slots; the driver reassembles fragments into a
kernel heap buffer; `STREAM_RECV` then copies the completed payload into the
posted userspace address. A verbs-only `RDMA_WRITE` envelope would retain those
copies and the RX worker and is therefore not a credible >=5% optimization.

A meaningful remote-placement path requires driver support: either DMA into a
pinned/dma-buf destination, or a mapped completion/data ring that removes heap
reassembly and the receive syscall. Such a path may expose `RDMA_WRITE` at the
verbs layer later. It must be capability-negotiated and opt-in so the existing
generic/Mellanox SEND/RECV path has no new hot-path cost.

## Promotion rules

- Prefer bytes, copies, fusion, ownership balance, overlap, and protocol
  changes over thread-count tuning.
- Keep all experimental paths default-off until two-node correctness and
  repeated end-to-end performance pass.
- Preserve the generic verbs path unchanged unless a negotiated capability is
  active; run the generic/Mellanox transport-policy test for every transport
  change.
- Synchronize binaries and providers explicitly on both nodes and verify their
  SHA-256 hashes before hardware A/B runs; the node filesystems are independent.
