# OdinLink node-two crash report — 2026-08-07

## Outcome

Node two suffered a real kernel panic, not an application crash. The panic was
triggered when node one unloaded OdinLink during recovery from a failed
handshake. That sent a logout to node two while Thunderbolt transmit
completions were still queued.

The affected driver let an in-flight transmit message keep only a raw pointer
to its stream. A concurrent stream close could free the stream and its wait
queue before the queued completion callback used that pointer. The saved crash
shows the callback trying to wake the freed wait queue.

No further inference or transport benchmark was started after this finding.

## Evidence

The initiating crash is preserved on node two in:

```text
/var/crash/202608072016/dmesg
/var/crash/202608072016/dump
```

The relevant sequence in the saved kernel log is:

```text
OdinLink: received logout from peer
thunderbolt ... hop deactivation failed
BUG: kernel NULL pointer dereference, address: 0000000000000000
workqueue: events ring_work [thunderbolt]
__wake_up
odl_tb5_tx_batch_callback [odl_tb5]
ring_work [thunderbolt]
```

The short crash-recovery boot that followed produced additional faults. Those
are secondary; the trace above is the initiating failure.

## Why our recovery action caused it

The attempted recovery unloaded the node-one module first. Module unload sent
a peer logout, and node two began tearing down the connection while a batch TX
callback was pending. The callback later dereferenced a stream that no longer
had a guaranteed lifetime.

Using the OdinLink data address for SSH was a separate operational weakness:
when the module disappeared, the management route disappeared too. An
independent management network preserves access, but by itself cannot prevent
this kernel race.

## Prevention implemented

| Layer | Change | Status |
| --- | --- | --- |
| Driver | Every latency and throughput TX message now takes a stream reference. The final callback or local error cleanup releases it. | Built for both node kernels; not loaded live |
| Operations | `scripts/odinlink-safe-reload.sh --reload` now always refuses. The script is check-only and requires an independent, non-OdinLink management route. | Active in the DS4 tree |
| DS4 transport | Explicit `--transport rdma` now fails closed if either rank lacks OdinLink/RDMA instead of silently using TCP. | Unit-tested |
| Bench harness | Both DSpark harnesses preflight `/dev/odl_tb5_0`, force RDMA on both ranks, and require an RDMA connection log afterward. | Implemented |

The driver fix is in:

```text
/home/wkljohn/Desktop/cc/OdinLink-Five/driver/odl_tb5_ring_dma.c
```

Because the nodes do not share files and currently run different kernels, the
same source was synchronized but the modules were built separately:

| Node | Kernel | Module SHA-256 |
| --- | --- | --- |
| Node one | `7.0.0-28-generic` | `743ef6cd87f159c6b624aa0d0e266fbf4a23aa2e3c7a006c36b6266a263b8872` |
| Node two | `7.0.0-29-generic` | `2f208bbc1074cd233ae3ff807d6fe17cd2a6bffc820eef950d02ecae5fe6710d` |

Both builds have source version `FDCC4AA4BBC03B9D04428E2`. They are on disk,
not deployed. Node one still has the older module (`F976E29176528118EF77424`)
resident from before this investigation. It has no device or clients, but must
not be live-unloaded; replace it through the controlled reboot below. The
compiler-name warning is cosmetic: both the kernel and module used GCC 15.2.0.

## Validation completed

- Both node-specific kernel modules compiled successfully.
- Node two remains reachable over Wi-Fi at `10.10.0.216`.
- Node two's crash dump remains preserved.
- Node two has no OdinLink module, device, DS4 process, or CLI process active.
- Node one has no OdinLink device, DS4 process, or CLI process active. Its old
  resident module was deliberately left untouched to avoid repeating the race.
- The unsafe reload option returns failure before making any module change.
- `make test-tp-hello` passes all transport-policy tests, including explicit
  RDMA rejection when either side lacks an RDMA device.

## Safe continuation

Do not use `rmmod odl_tb5` or PCI-driver unbind as a link-reset mechanism. A
Thunderbolt controller unbind can itself block inside the kernel and make SSH
unresponsive even over an independent network. Until the patched driver has
passed teardown stress testing, recover a wedged link only by rebooting in a
controlled maintenance window:

1. Stop DS4 and OdinLink clients on both nodes.
2. Confirm independent management access in both directions.
3. Confirm each node's module `vermagic` matches its running kernel.
4. Reboot both nodes; do not live-unload the old module.
5. Load the locally built patched module once on each quiet node.
6. Run standalone OdinLink lifecycle and byte-integrity stress tests first.
7. Resume DS4 only after both `/dev/odl_tb5_0` devices are present and the
   standalone transport tests pass.

If teardown stress still triggers a panic, keep live reload disabled and treat
the driver as requiring reboot-only recovery while the remaining ring shutdown
ordering is investigated.
