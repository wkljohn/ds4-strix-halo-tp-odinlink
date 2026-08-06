# Q8-to-F16 cache policy

The ROCm Q8-to-F16 weight cache is disabled by default in this fork. Ordinary
inference and benchmark commands use the direct Q8 kernels and do not allocate
the cache.

The policy follows a matched TP=2 OdinLink measurement on 2026-08-06:

| Historical configuration | Cached per rank | Prefill | Decode |
|---|---:|---:|---:|
| Former automatic cache | 9.85--9.91 GiB | 115.56 t/s | 13.80 t/s |
| Former direct Q8 | 0 GiB | 87.67 t/s | 13.79 t/s |

The cache materially accelerated this prefill workload but did not improve the
300-token decode result beyond measurement noise. With the cache present,
ROCm stopped allocation with only 4.81--4.86 GiB of its reported 96 GiB
aperture free, and external monitoring could report usage near 99%.

After changing the default, a separate 10,093-byte prompt completed at 101.32
prefill t/s and 13.33 decode t/s with 30 generated tokens. Neither rank printed
the cache warning or budget message. An explicit opt-in startup check printed
the warning on both ranks and reproduced the 9.85/9.91 GiB allocations.

The current compact-Q8 kernel reuses each Q8 attention-output weight block
across a 16-token prompt tile. On the same 10,093-byte, 30-generation workload
it reached 138.97 prefill t/s and 13.32 decode t/s with no persistent cache;
generated output was byte-identical to the archived 101.32/13.33 direct-Q8
run. A same-binary one-token A/B measured 136.18 t/s by default and 97.50 t/s
with `DS4_ROCM_ATTN_OUT_Q8_A_PREQ_TOKTILE=0` (+39.7%).

Users may make the memory/prefill trade explicitly:

```sh
export DS4_ROCM_ENABLE_Q8_F16_CACHE=1
```

Only the exact value `1` enables it. DS4 prints a warning stating the expected
memory cost. Leaving the variable unset is the supported default for inference
and testing. The older `DS4_CUDA_NO_Q8_F16_CACHE` kill switch remains accepted
and takes precedence, but it is no longer needed in normal commands.
