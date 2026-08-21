# TP completion-word ordering oracle

Date: 2026-08-21

## Purpose

The exact Q4_K FFN output-row balance oracle already proved the arithmetic, but
the previous production attempt lost its gain to a third `MOE_MID` host
callback. A callback-free design needs a safe ordering primitive between:

1. peer payload reception;
2. a host-published completion word;
3. a GPU consumer that reads the peer payload.

`tests/test_tp_completion_ordering.cu` reuses DS4's real TP handshake, selected
verbs provider, QP, registered slab, decode receive window, and SEND/RECV gate.
After the receive CQE, a transport thread release-stores a monotonically
increasing completion word. A GPU kernel already waiting on that word verifies
all 4,096 F32 payload values. The watchdog uses the same host-to-GPU direction
and fails closed rather than leaving an unbounded GPU poll.

This does not modify production scheduling or transport.

## Rejected allocation

Publishing the CPU word inside OdinLink's production `hipMalloc` slab timed out
on both ranks even though the RDMA exchange completed. A CPU store to ordinary
`hipMalloc` is therefore not a valid prompt GPU notification mechanism on this
gfx1151 stack.

The accepted oracle allocates a separate 64-byte `hipHostMallocMapped`
completion page. The payload remains in each provider's production allocator:
mapped host memory for mlx5 and `hipMalloc` for OdinLink. This adds no model
cache and only a cache-line-sized persistent control allocation if integrated.

## Sustained result

| Provider | Payload allocator | Iterations per rank | Transport errors | Word timeouts | Stale payloads | Elapsed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ConnectX-4 Lx RoCE v2 RC | `hipHostMallocMapped` | 25,800 | 0 | 0 | 0 | 2.264 s |
| OdinLink RC/SEND | `hipMalloc` | 25,800 | 0 | 0 | 0 | 3.482 s |

OdinLink reported 25,800 WC-stream calls and 422,707,200 bytes per rank with
zero provider fallbacks. Full logs are in this directory's ignored
`ordering-probe/` folder on node 1; the worker logs were copied back from node
2 because the filesystems are independent.

## What this proves—and does not prove

It proves that `receive CQE -> CPU release store in mapped control memory -> GPU
poll -> payload read` remained ordered and visible through 25,800 real gate
exchanges on both providers. It avoids the previously unreliable inverse
direction (`GPU stream signal -> host`).

It does not yet prove that a background producer-event queue and one consumer
wait can replace the rejected third callback profitably. Production integration
must remain default-off, negotiate an exact TP feature bit, retain a bounded
timeout/fail-closed path, and pass the established Q4_K/Q2_K full fingerprints
on both providers before commit or default consideration.
