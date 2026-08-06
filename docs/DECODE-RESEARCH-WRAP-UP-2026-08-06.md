# Decode research continuation wrap-up — 2026-08-06

This checkpoint closes the current decode-research continuation. It does not
claim that decode can never improve; it records which measured directions are
retained or ruled out before new information changes the search space.

## Retained acceleration

- Original OdinLink provider baseline: 9.96 decode t/s.
- Packed gfx1151 Q8 attention-output low projection: 11.15 to 11.78 t/s in
  the paired test (+5.7%).
- Packed Q8 attention-output expansion: 11.93 to 12.93 t/s (+8.4%).
- Packed Q8 Q-B projection: 12.72 to 13.34 t/s in the profiled pair; the
  unprofiled 1,000-token stability run reached 13.48 t/s.
- The current low-memory default completed a matched 300-token control at
  13.83 t/s. The formerly automatic 9.85--9.91 GiB Q8-to-F16 cache did not
  improve measured decode and is now warning-bearing opt-in only.

## Q4_K compute conclusion

The routed-MoE Q4_K path measured 0.281/0.247 ms for gate/up and 0.113/0.106
ms for down on rank 0/rank 1. LDS activation staging regressed, 128-thread
geometry was neutral/asymmetric, and 512 threads regressed. The weight stream
is already bandwidth dominated, so no isolated remaining Q4_K kernel change
has a credible 5% whole-token ceiling.

Q4_K weights remain packed four-bit storage. The active tile is unpacked into
INT8 operands in LDS/registers and the compiled gfx1151 code object contains
native `v_wmma_i32_16x16x16_iu8` instructions. Original Q4_K scales and minima
are retained; there is no persistent expanded weight copy.

## Transport conclusion

Detailed gate profiling put lock, post, and replenish near 3.5 us/gate, local
send CQEs at 6--10 us, and peer arrival near 48--62 us before rank-compute
skew. Selective signalling, work-request reuse, or a provider-only software
`RDMA_WRITE` wrapper cannot remove enough of the measured path for 5% end to
end. OdinLink's current receive path still reassembles in kernel memory and
copies through `STREAM_RECV`; meaningful RDMA WRITE requires driver-level
direct placement or a mapped completion/data ring.

Raising the provider caller-thread inline ceiling from 4 KiB to 16 KiB removed
the queued-copy counters but regressed decode from 13.76 to 11.75 t/s. The
provider worker is supplying useful overlap.

Producer-ready attention chunking was implemented with explicit GPU completion
signals and two 8 KiB messages. It was byte-correct but changed 13.83 to 13.74
t/s while provider calls rose from 28,036 to 53,750. That implementation was
removed. Its measured message overhead also rules out extending the same
design into the more complex Q4_K FFN producer as a likely 5% gain.

## Deferred, not implemented

A true driver direct-placement path remains technically possible and the user
has authorized building RDMA WRITE if future evidence makes it likely to help.
Current measurements do not meet that bar: a verbs-only implementation keeps
the dominant copies, while a driver redesign has a high effort/risk ceiling
and is unlikely to recover the entire arrival floor. It must remain capability
negotiated so the generic/Mellanox SEND/RECV path has no added hot-path cost.

All raw two-node logs are retained under the Git-ignored
`research-results/2026-08-06/raw/` tree. The next decode investigation should
start from new evidence supplied after this checkpoint rather than repeating
the rejected candidates above.
