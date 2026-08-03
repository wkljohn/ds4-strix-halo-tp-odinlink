# The gate primitive is fine. The GPU never reaches the first gate.

Supersedes the signal-memory hypothesis in docs/GATE-DEADLOCK.md, which
measurement REFUTED.

## T3 probe: both signal directions work (scripts/t3_gate_signal_probe.cpp)

    canUseStreamWaitValue = 1
    hipMallocSignalMemory UNAVAILABLE -> falling back to hipHostMalloc
    host ptr=0x7bddafcae000  device ptr=0x7bddafcae000  (identical)
    A. GPU stream-write -> CPU poll : SEEN (0.000 s)
    B. CPU store -> wait released   : RELEASED (0.000 s)

So on this UMA APU the mapped-host pointer IS the device pointer, and passing
the host pointer to `hipStreamWaitValue64` (which ds4_rocm.cu:407 does) is
correct here. The signal-memory fallback is sound. **My hypothesis was wrong.**

Also explains the long-standing sticky "invalid argument": it is
`hipExtMallocWithFlags(hipMallocSignalMemory)` failing at ds4_rocm.cu:337, which
leaves the error latched for the next `cudaGetLastError()` to misreport. Benign.

## The trace (DS4_TP_TRACE=1, both ranks)

    [tp] encode ch=1 seq=1 kind=2 -> enqueue
    ... through seq=44 ...
    total [tp] lines: 44        (coordinator AND worker, identically)

**44 encodes, 0 pumps.** Reading that:

- 44 = one forward pass of big gates (43 layers + 1). The host enqueued a whole
  pass without blocking, which is CORRECT: `hipStreamWaitValue64` blocks the
  stream, not the caller.
- The service thread never saw `gpu_flag >= 1` even once.
- Thread states: main R (spin), service R (spin), one thread S in
  `kfd_wait_on_events` (blocked on the GPU), network thread S idle.
- The socket carried 204 bytes sent / 92 received - the hello, then nothing.

If the GPU had executed the first `hipStreamWriteValue64`, the service thread
would have pumped (T3 shows that path works in 0.000 s). It did not. So **the
GPU is stuck in a kernel that precedes the first gate**, holding it at 100%.
The gate handshake is downstream of the real fault and never got a chance to run.

## Where to look next

The prime suspect is patch 12's own change, since the hang appeared with it:
the launcher now runs with `n_total_expert = 128` while the router still
produces selections over the full 256. If any consumer of `selected` was NOT
rebased - a bucket/count/sort kernel sized by `n_total_expert`, or the
shared-expert path - an index of 128..255 against a 128-entry structure is an
out-of-bounds access, which on GPU can spin rather than fault.

Concretely, next steps:
1. Find every consumer of `selected` downstream of the remap and confirm each
   agrees on the expert count. `routed_moe_build_plan` sizes its counts array by
   `n_total_expert`; check the router/bucket kernels use the same value.
2. Bisect cheaply: set the shard range to the FULL range (lo=0, hi=n_total) so
   the remap is identity. If the hang disappears, it is the remap, not TP.
3. Only then re-check the gate.

Note the earlier per-2s VRAM sampling showed patch 12 DID fix the residency leak
(95.52 -> 85.65 GiB peak, no arena OOM), so that part stands regardless.
