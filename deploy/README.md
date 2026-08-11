# 256K TP=2 deployment behind Caddy

This profile runs one cache-free Q4_K target across two Strix Halo nodes,
keeps the optional Q8 DSpark drafter on rank 0, and exposes DS4 only on
`127.0.0.1:8090`. Caddy remains responsible for TLS, authentication, and the
public hostname.

```sh
make strix-halo
cp deploy/config.env.example deploy/config.env.local
$EDITOR deploy/config.env.local
deploy/ds4-tp-caddy.sh start
deploy/ds4-tp-caddy.sh status
```

The launcher starts both independent model loads together, requires readable
OdinLink devices, validates matching binary and sampled model fingerprints,
and refuses to replace unrelated processes. It uses a 262,144-token context
with a memory-safe 2,048-token prefill workspace,
the server's single resident session, the benchmarked 46/54 expert split, and
explicit RDMA. Native batched-session mode is intentionally not enabled because
it disables speculative decoding. Profiling and DSpark statistics are also off
in production.

On the 96 GiB reference nodes this full profile settles near 97--98% reported
VRAM use. That is expected but leaves little safety margin: do not add resident
sessions or enable the Q8-to-FP16 weight cache. If another GPU workload must
share either node, stop this service or lower `CONTEXT` first.

Both nodes need the repository, target GGUF, drafter GGUF, and OdinLink
provider at the absolute paths configured in `config.env.local`; their
filesystems are not assumed to be shared. Logs are retained under
`research-results/deployment/` and are excluded from Git.

Useful commands:

```sh
deploy/ds4-tp-caddy.sh logs
deploy/ds4-tp-caddy.sh stop
caddy validate --config /etc/caddy/Caddyfile
```

The local Caddy route should proxy to `127.0.0.1:8090` with streaming flush
enabled and a response timeout long enough for initial model loading. Keep
port 8090 loopback-only; do not bypass Caddy's authentication at the firewall.
