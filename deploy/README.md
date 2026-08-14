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
`research-results/deployment/` and are excluded from Git.

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
