# Q8-to-F16 cache policy

The ROCm Q8-to-F16 weight cache is disabled by default in this fork. Ordinary
inference and benchmark commands use the direct Q8 kernels and do not allocate
the cache.

The policy follows a matched TP=2 OdinLink measurement on 2026-08-06:

| Configuration | Cached per rank | Prefill | Decode |
|---|---:|---:|---:|
| Former automatic cache | 9.85--9.91 GiB | 115.56 t/s | 13.80 t/s |
| Default direct Q8 | 0 GiB | 87.67 t/s | 13.79 t/s |

The cache materially accelerated this prefill workload but did not improve the
300-token decode result beyond measurement noise. With the cache present,
ROCm stopped allocation with only 4.81--4.86 GiB of its reported 96 GiB
aperture free, and external monitoring could report usage near 99%.

After changing the default, a separate 10,093-byte prompt completed at 101.32
prefill t/s and 13.33 decode t/s with 30 generated tokens. Neither rank printed
the cache warning or budget message. An explicit opt-in startup check printed
the warning on both ranks and reproduced the 9.85/9.91 GiB allocations.

Users may make the memory/perfill trade explicitly:

```sh
export DS4_ROCM_ENABLE_Q8_F16_CACHE=1
```

Only the exact value `1` enables it. DS4 prints a warning stating the expected
memory cost. Leaving the variable unset is the supported default for inference
and testing. The older `DS4_CUDA_NO_Q8_F16_CACHE` kill switch remains accepted
and takes precedence, but it is no longer needed in normal commands.
