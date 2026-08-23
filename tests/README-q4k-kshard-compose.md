# Q4_K K-shard rank-1 composition oracle

This gfx1151-only H7.5 test is deliberately separate from the H7 residency
commit. It installs the rank-1 all-expert K-half through the atomic installer,
runs the default-off one-token packed primitive with a nonzero shared-expert
addend, and compares its output bit-for-bit with an independently CPU-packed
compact rank-1 reference using the shipped ordinary Q4_K primitive. TP is
shut down before the reference and ownership filtering is suspended only in
the oracle, without destroying or recreating the GPU runtime or tensors.

It has no production caller, transport, prefill, default-path, or performance
claim. H8 cannot start until this oracle passes three independent processes.
