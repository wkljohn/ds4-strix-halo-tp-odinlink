# DSpark diverse promotion gate

DSpark candidates must survive the same compact cross-disciplinary 4,096+300
workload used by ordinary inference. It covers software debugging, quantitative
science, policy retrieval, and structured-data analysis in one model load. The
screen supplements the fixed production benchmark; it does not replace isolated
verifier tests or the three-run production median.

Run three frozen DSpark baselines, one candidate, and three same-binary ordinary
controls. Candidate validation mode is mandatory, so short generation,
fingerprint, semantic, transport, or RDMA failures stop the run.

```sh
export DS4_RESEARCH_ROOT=/absolute/path/to/research-results
export MODEL=/absolute/path/to/DeepSeek-V4-Flash-Q4_K.gguf
export MTP=/absolute/path/to/dspark-Q8_0.gguf
export DS4_BENCH_CANDIDATE=1
export DS4_BENCH_EXPECT_FNV64="$DSPARK_FNV64"

DS4_BENCH_DSPARK=1 DS4_BENCH_MTP="$MTP" \
  ./scripts/run-tp-diverse-bench.sh dspark-candidate "$MODEL"

export DS4_BENCH_EXPECT_FNV64="$ORDINARY_FNV64"
./scripts/run-tp-diverse-bench.sh ordinary-control-r1 "$MODEL"
./scripts/run-tp-diverse-bench.sh ordinary-control-r2 "$MODEL"
./scripts/run-tp-diverse-bench.sh ordinary-control-r3 "$MODEL"
```

Create the immutable summary after supplying the three corresponding frozen
DSpark baseline CSVs:

```sh
./scripts/diverse-bench-gate.py create --mode dspark --lane A \
  --baseline "$DS4_RESEARCH_ROOT/bench-runs/dspark-base-r1.csv" \
  --baseline "$DS4_RESEARCH_ROOT/bench-runs/dspark-base-r2.csv" \
  --baseline "$DS4_RESEARCH_ROOT/bench-runs/dspark-base-r3.csv" \
  --ordinary "$DS4_RESEARCH_ROOT/bench-runs/ordinary-control-r1.csv" \
  --ordinary "$DS4_RESEARCH_ROOT/bench-runs/ordinary-control-r2.csv" \
  --ordinary "$DS4_RESEARCH_ROOT/bench-runs/ordinary-control-r3.csv" \
  --candidate "$DS4_RESEARCH_ROOT/bench-runs/dspark-candidate.csv" \
  --output "$DS4_RESEARCH_ROOT/candidates/NAME/diverse-summary.json"
```

The summary records and hashes every CSV and manifest. DSpark evidence cannot
be substituted with an ordinary-mode summary. Lane A additionally requires the
candidate token fingerprint to match the frozen DSpark baseline. Lane B/C
numerical changes still follow the promotion policy in
`$DS4_RESEARCH_ROOT/policies/GATE-PROMOTION.md`.
