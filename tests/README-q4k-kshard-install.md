# Q4_K K-shard atomic installer oracle

This gfx1151-only synthetic test proves the default-off residency transition
needed before production K-shard dispatch. It loads all experts with one
1024-row gate/up half and the paired four-block down-K half, then verifies
exact rank windows, all-expert addressability, remap suspension/restoration,
linear full-tensor refusal after success, failure, and release, idempotence,
and failure/cleanup atomicity. Empty dense-span lists fail closed rather than
mapping the entire GGUF. A failed or released research install leaves a small
routed-range tombstone until the next atomic install or model cleanup, so an
ordinary full-expert allocation cannot silently create a second residency.

It does not modify engine startup, add a production caller, authorize packed
prefill, negotiate transport, or promote a performance result. Ordinary TP
expert-id sharding remains unchanged unless this inert API is called.
