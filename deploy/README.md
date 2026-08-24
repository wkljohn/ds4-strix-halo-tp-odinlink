# 256K TP=2 deployment behind Caddy

This profile runs one cache-free Q4_K or hybrid-Q2 target across two Strix Halo
nodes over OdinLink or mlx5 RoCE v2, uses exact ordinary decode by default,
and exposes DS4 only on
`127.0.0.1:8090`. Caddy remains responsible for TLS, authentication, and the
public hostname.

```sh
make strix-halo
cp deploy/config.env.example deploy/config.env.local
$EDITOR deploy/config.env.local
deploy/ds4-tp-caddy.sh start
deploy/ds4-tp-caddy.sh status
```

After the final build, set `DS4_SERVER_SHA256` in `config.env.local` from
`sha256sum ./ds4-server`. Startup fails closed if that pin does not match, so a
stale coordinator cannot use a different TP gate schedule from the worker.

The launcher starts both independent model loads together, validates the
selected RDMA provider plus matching binary and sampled model fingerprints,
and refuses to replace unrelated processes. It uses a 262,144-token context
with a memory-safe 2,048-token prefill workspace,
the server's single resident session, the balanced 50/50 ordinary-decode expert
split, and explicit RDMA. DSpark uses 46/54 only when explicitly enabled.
Native batched-session mode is intentionally not enabled because
it changes the execution profile. Profiling is off in production. The launcher
inspects routed-expert quantization and selects the same safe Q4_K or Q2
prefill defaults enforced by the pre-main benchmark gate. Set
`DSPARK=1` only for experimental testing; the launcher warns because the
current TP verifier does not match the target-only token fingerprint.
Ordinary Q4_K/Q2_K launches also select the validated ordered ROCm TP callback
and temporal-compressor schedule, the RoCE prefill wavefront, and the
shape-gated M256/K128 Q8 projection, exact cooperative HC decode stage, and
the exact long-context indexer radix tree. The wavefront provider-gates itself
off for OdinLink, and the radix tree engages only after 8,192 compressed rows.
Set `PREFILL_FFN_WAVEFRONT=0`, `Q8_M256_K128=0`,
`HC_STAGE_EXACT_COOP=0`, or `INDEXER_TOPK_RADIX_TREE=0` in the deployment
config for a symmetric rollback; no extra shell environment is required.
Research branches can also forward the symmetric
`Q4K_KSHARD_RESEARCH` and `ATTN_DECODE_SEQTILE_RESEARCH` switches. They remain
off by default and must not be enabled without their matching gate dossier.

On the 96 GiB reference nodes the optional resident DSpark profile settles near
97--98% reported VRAM use and leaves little safety margin. Ordinary production
decode has more headroom. Do not add resident sessions or enable the
Q8-to-FP16 weight cache without rechecking memory. If another GPU workload must
share either node, stop this service or lower `CONTEXT` first.

Both nodes need the repository and target GGUF at the absolute paths configured
in `config.env.local`; the OdinLink profile also needs its provider.
Experimental DSpark additionally
requires the drafter GGUF on both nodes. Their filesystems are not assumed to
be shared. Logs are retained under
`$DS4_RESEARCH_ROOT/deployment/` and are excluded from the source branches.

Set `RDMA_PROFILE=roce-v2`, `COORDINATOR_RDMA_ADDR=192.168.99.1`,
`LOCAL_RDMA_DEVICE=mlx5_0`, `PEER_RDMA_DEVICE=mlx5_1`, and
`RDMA_GID_INDEX=3` for Mellanox. `ODINLINK_ROOT` is then unnecessary.
RoCE also requires passwordless `sudo` for the launcher: the coordinator runs
as the system transient service `ds4-tp-coordinator.service` with an actually
unlimited locked-memory limit. The launcher verifies that effective process
limit and checks the peer's SSH-session limit before model loading. OdinLink
retains its transient user service and does not acquire this requirement.

Useful commands:

```sh
deploy/ds4-tp-caddy.sh logs
deploy/ds4-tp-caddy.sh stop
caddy validate --config /etc/caddy/Caddyfile
```

The local Caddy route should proxy to `127.0.0.1:8090` with streaming flush
enabled and a response timeout long enough for initial model loading. Keep
port 8090 loopback-only; do not bypass Caddy's authentication at the firewall.

## Agent-client completion limits

Set an explicit `max_tokens` or `max_completion_tokens` in agent clients. A
4,096–8,192 token limit is a practical starting point. The server repairs a
simple unterminated DSML tool call when generation reaches the request limit;
a 32,000-token client limit can therefore spend tens of minutes decoding a
malformed tool argument before the repair runs. This is a model-output/API
failure mode, not evidence that RDMA stopped making progress. Health checks
remain responsive after the affected request completes.
