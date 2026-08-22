# OpenAI-compatible TP server benchmark

This repository has two complementary benchmark layers:

- `run-tp-ds4-bench.sh` launches both ranks itself, requires RDMA, validates
  binary/model agreement and deterministic fingerprints, and remains the
  correctness gate for optimization claims.
- `scripts/run-api-bench.sh` measures client-visible prompt processing,
  generation, and time-to-first-token through the running OpenAI-compatible
  server. It follows the `llama-benchy` methodology used by
  [ds4-on-spark](https://github.com/Entrpi/ds4-on-spark).

The API benchmark pins
[`llama-benchy`](https://github.com/eugr/llama-benchy) commit
`e9be344578cec17745066b220798b80a0d2686d3`. Its default matrix is:

```text
prompt processing:  2,048 tokens
generation:         32, 128, and 300 tokens
context depth:      0, 4,096, and 16,384 tokens
measured runs:      3 per shape
discarded warmups:  1 per shape
concurrency:        1
latency mode:       generation
prefix avoidance:   enabled
```

Use a dedicated server instance. DS4 currently keeps one resident session;
the sweep deliberately replaces it and therefore refuses to start without an
explicit confirmation. Do not point it at an interactive production session.

```sh
scripts/run-api-bench.sh --confirm-dedicated-server
```

The default endpoint is `http://127.0.0.1:8090/v1`. Common overrides:

```sh
scripts/run-api-bench.sh --confirm-dedicated-server \
  --base-url http://127.0.0.1:8090/v1 \
  --model deepseek-v4-flash \
  --pp 2048 \
  --tg 32 128 300 \
  --depth 0 4096 16384 65536 \
  --runs 3 \
  --out "$DS4_RESEARCH_ROOT/api-bench/q4-roce.csv"
```

Use `--api-key` when benchmarking through an authenticated proxy. The key is
never written to the metadata sidecar or printed by `--dry-run`.

The default output is an ignored CSV under `$DS4_RESEARCH_ROOT/api-bench/`, plus
a `.meta.txt` sidecar containing the DS4 commit, pinned benchmark version, and
test matrix. Review and explicitly add selected result files when publishing
them.

Interpret these results as serving/API measurements. They include request
rendering, streaming, and benchmark-tool latency estimation and are not
directly interchangeable with the internal fixed-workload `ds4-bench-tp`
numbers in the main README.
