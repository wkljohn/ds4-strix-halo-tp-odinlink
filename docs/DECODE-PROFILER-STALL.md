# ROCm TP=2 RDMA decode profiling stalls at row-gate FFN exchange (2nd/3rd token)

Following DECODE-ACCELERATION-PLAN.md's Stage 0 procedure (symmetric,
`DS4_ROCM_DECODE_STAGE_PROFILE=1 DS4_ROCM_DECODE_STAGE_PROFILE_LAYER=20
DS4_ROCM_MOE_DECODE_PROFILE=1` on both ranks, OdinLink RDMA TP=2, DS4-Flash
Q4_K, `-c 2048 --temp 0`) fails before reaching the plan's required >=500
tokens:

    ds4-tp: timeout waiting gate seq 128 (recv_done 127)
    ds4: ROCm TP exchange failed (kind 0 layer 20 seq 128); releasing anyway so the GPU does not hang
    ds4: metal layer stage part=decode layer=20 pos=43 tokens=1 ffn_hc_post=304747.509 ms
    ds4: decode failed: rocm decode failed

**This is much earlier than it looks.** `pos=43` is KV position (prompt +
generated), not generated-token count. Row-gate `seq` is global/monotonic for
the whole TP session (`ds4_rocm.cu:428-435`, one counter per channel), the
schedule is 86 row gates/token for this model (`ds4_tp.c:130-134`), and slot
= `layer*2 + gate` (`ds4_tp.c:1032-1046`). So **seq 128 = 86 + 20*2 + 2 = the
SECOND decode token's layer-20 FFN gate.** A second run without
`DS4_TP_BIG_DIRECT` failed at seq 214 = the third token's same gate - same
failure, ~1 token later, not two different bugs.

## Ruled out

- **Not DS4_TP_BIG_DIRECT.** A/B with and without it: fails at seq 128 vs 214,
  same signature either way. Code-level, `tp_big_out`/`tp_big_in` only alias
  `batch_routed_out`/`batch_ffn_out` (prefill tensors, ds4.c:57084-57116);
  decode's per-token path never references them.
- **Not a receive-window/CQ/ring-wrap size effect.** Receive window is a
  compile-time constant of 16 (`ds4_tp.c:137`); QP allows 64 recv WRs, 256
  send WRs, CQ has 512 entries (`ds4_tp.c:815-833`). 128 and 214 don't align
  with any of these boundaries, and both are far too early (2nd/3rd token)
  for a window-exhaustion story anyway.
- **Not a simple "profiling makes it slower, raise the timeout" issue.**
  Default `DS4_TP_TIMEOUT_SEC` is already 300s (`ds4_tp.c:50-52`) against
  sub-millisecond normal per-stage times. A single gate legitimately waiting
  five minutes is not "slow," it's stuck.
- **Not a gate-topology change.** Both ranks still encode exactly one ATTN
  gate and one FFN gate per layer regardless of profiling
  (`ds4.c:22675-22686`, `ds4.c:23947-23969`).
- One correction to the plan doc's own claim: `DS4_ROCM_DECODE_STAGE_PROFILE`
  (the env var) does NOT actually disable the fused/overlap paths gated on
  `g->decode_stage_profile` at `ds4.c:22946-22960,23231-23247` for this ROCm
  path - that flag is set from a Metal-side variable, not the ROCm env var
  (`ds4.c:16826-16830`). The selected/shared overlap paths are unavailable
  regardless (`tp_world<2` required, `ds4.c:23286-23305`), so this doesn't
  change the failure, just corrects the mechanism.

## Leading hypothesis (unresolved)

The stage profiler inserts a full `cudaDeviceSynchronize()` at every profiled
boundary (`ds4.c:21558-21564,26668-26688`, `ds4_rocm_runtime.cuh:6072-6075`);
`DS4_ROCM_MOE_DECODE_PROFILE` separately synchronizes on GPU events around
every MoE op (`ds4_rocm_moe_launch.cuh:203-279,2144-2153,2460-2463`). At the
failing point: stage sync after `routed_moe_folded`, then the FFN gate is
encoded, then the NEXT boundary waits for that gate and charges the 300s wait
to `ffn_hc_post` (`ds4.c:23935-24001`) - exactly matching the log. Best
current guess: this dense synchronization pattern exposes a ROCm
stream/gate-service progress failure - one rank stops reaching/publishing its
next FFN gate while the other has already sent and is waiting. Not confirmed:
transport message loss and worker-side output/scheduler starvation are weaker
alternative explanations neither confirmed nor ruled out by static reading.

## Workaround (use this, don't chase timeout/window tuning)

**Stage 0b** (already in DECODE-ACCELERATION-PLAN.md as the "less invasive"
alternative): instrument the service thread's three intervals separately
(GPU-arrival detection, transport callback duration, release-to-next-arrival)
instead of the full stage profiler. This avoids both the repeated
`cudaDeviceSynchronize()` boundaries and the MoE event waits that appear to
cause the stall, and it directly measures the suspected progress-failure
mechanism. Not yet implemented - requires new instrumentation in the service
thread, not just an env var.

Cheapest immediate thing to try first (config-only, no code change):
`DS4_ROCM_MOE_DECODE_PROFILE=1` ALONE, `DS4_ROCM_DECODE_STAGE_PROFILE` unset -
keeps the MoE event syncs but removes the per-stage full-device boundaries.
Gives MoE decomposition only (not the broad attn/router/FFN split Stage 0
wants), and may still hit the same stall if the MoE-side syncs alone are
sufficient to trigger it - untested at time of writing.

## Status

Blocking: Stage 0's mandated >=500-token symmetric decode profile has not
been obtained. Filed 2026-08-05 per Codex root-cause investigation (see
memory for the full analysis). Next diagnostic: gate tracing on both ranks to
see which side actually stalls (encode-side never publishing vs
consume-side never returning), or implement Stage 0b.
