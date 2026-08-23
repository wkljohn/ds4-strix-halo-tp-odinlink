# Q4_K packed-slice registry oracle

```sh
make test-rocm-q4k-packed-slice-registry
```

This standalone ROCm oracle validates the nonlinear weight layout required by
the gate-free routed-expert K-shard experiment. It loads deterministic Q4_K
bytes through the production model-fd staging path and checks two layouts:

- a contiguous 256-row output half with complete Q4_K rows;
- four contiguous Q4_K blocks selected from every down-projection row.

Device readback must match a separate host pack byte-for-byte and reproduce
the frozen hashes `eee496fd886b6b83` and `662be17fcaa0e383`. Eight experts
force the four-buffer asynchronous fd staging ring to wrap. The oracle also
checks the mmap/no-fd pack path, persistent-byte accounting, descriptor
lifetime across a model-map transition, declaration-after-cache refusal,
idempotent and conflicting declarations, geometry/alignment rejection,
disjoint linear-range success, and fail-closed legacy range/span lookup.

This test validates residency plumbing only. It does not enable full-model
K-sharding, change ordinary inference, establish numerical equivalence, or
authorize a performance candidate.
