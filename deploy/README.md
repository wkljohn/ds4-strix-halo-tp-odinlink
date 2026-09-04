# 256K TP=2 deployment behind Caddy

This profile runs one cache-free Q4_K or hybrid-Q2 target across two Strix Halo
nodes over OdinLink or mlx5 RoCE v2, uses exact ordinary decode by default,
and exposes DS4 only on
`127.0.0.1:8090`. Caddy remains responsible for TLS, authentication, and the
public hostname.

```sh
DS4_ROCM_HOME=/absolute/path/to/rocm-7.14.0 \
  make -j"$(nproc)" strix-halo
cp deploy/config.env.example deploy/config.env.local
sed -i "s/^DS4_SERVER_SHA256=.*/DS4_SERVER_SHA256=$(sha256sum ./ds4-server | awk '{print $1}')/" \
  deploy/config.env.local
$EDITOR deploy/config.env.local
deploy/ds4-tp-caddy.sh start
deploy/ds4-tp-caddy.sh status
```

`DS4_ROCM_HOME` must name the ROCm 7.14 installation root containing
`bin/hipcc`. The `sed` command pins the final `ds4-server` build in
`config.env.local`. Startup fails closed if that pin does not match, so a stale
coordinator cannot use a different TP gate schedule from the worker.

Run the launcher on the coordinator. It opens an SSH session from the
coordinator to `PEER_MGMT`; install that coordinator user's public key on the
worker and verify `ssh -o BatchMode=yes` succeeds first. Keep `PEER_MGMT` on an
independent NetworkManager-managed Ethernet or Wi-Fi address, not an OdinLink
or Thunderbolt address.

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
The validated Q4_K K-shard and contiguous attention sequence tile are enabled
by default. Set `Q4K_KSHARD_RESEARCH=0` or
`ATTN_DECODE_SEQTILE_RESEARCH=0` in the deployment config for a symmetric
rollback on both ranks.

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
`RDMA_GID_INDEX=3` for Mellanox RoCE v2. `ODINLINK_ROOT` is then unnecessary.
For a direct Mellanox InfiniBand cable (ConnectX-3, `mlx4`), set
`RDMA_PROFILE=ib-mlx4`, the coordinator's InfiniBand interface address as
`COORDINATOR_RDMA_ADDR`, `LOCAL_RDMA_DEVICE`/`PEER_RDMA_DEVICE` from
`ibv_devinfo -l` on each node, and `RDMA_GID_INDEX=0`. A subnet manager
(`opensm`) must run on one node so both ports get LIDs; the launcher checks
link layer, ACTIVE state, and LID on both sides before model loading.
Mellanox RDMA (RoCE v2 or InfiniBand) also requires passwordless `sudo` for
the launcher: the coordinator runs as the system transient service
`ds4-tp-coordinator.service` with an actually unlimited locked-memory limit.
The launcher verifies that effective process limit and checks the peer's
SSH-session limit before model loading. OdinLink retains its transient user
service and does not acquire this requirement.

Set `DS4_TP_CONTAINER=1` (plus the image reference, content-addressed image ID, dedicated
read-only artifact volume, separate writable runtime volume, and optional init command)
to run both ranks in podman containers instead, for hosts where ROCm exists
only inside a container image. The memlock requirement then moves into the
container (`--ulimit memlock=-1`), the coordinator runs as a `systemd --user`
transient unit supervising a foreground `podman run`, the peer worker is a
detached `--rm` container, and passwordless `sudo` is not required. The
launcher probes the image, GPU/InfiniBand device access, and the memlock
limit inside a throwaway container on both nodes before model loading. It
refuses mutable tags, mismatched image IDs, broad/home/credential directories,
and missing peer volumes. Container mode is a dedicated-host, administrator-
only trust boundary: host networking/IPC and `/dev/kfd` remain necessary for
this TP/RDMA design.

Before enabling it, install the exact same image on both nodes and record its
content identity:

```sh
podman build --pull=never -f deploy/Containerfile.rocm714-rdma \
  -t localhost/ds4-rocm714-rdma:local .
podman image inspect --format '{{.Id}} {{.Digest}} {{join .RepoDigests " "}}' \
  localhost/ds4-rocm714-rdma:local
```

The supplied Containerfile pins the ROCm 7.14/gfx1151 base-image digest and
installs matching, version-pinned Ubuntu `libibverbs` providers and inspection
utilities. Do not bind-mount provider libraries from the host: a host/container
ABI mismatch makes verbs discovery fail and may not be caught until after the
large model has loaded.

Set `DS4_TP_CONTAINER_IMAGE` to that local tag (or a registry digest) and
`DS4_TP_CONTAINER_IMAGE_ID` to the 64-hex `.Id` returned on both nodes. Do not
use a tag that was built independently on each machine. When an image is
transferred with `podman save|load`, the manifest digest may be rewritten; the
content-addressed image ID is therefore the mandatory parity pin. Use a dedicated pair
of directories such as `/srv/ds4-container` and `/srv/ds4-container-runtime`;
the launcher mounts artifacts read-only and runtime state separately.

Useful commands:

```sh
deploy/ds4-tp-caddy.sh logs
deploy/ds4-tp-caddy.sh stop
caddy validate --config /etc/caddy/Caddyfile
```

The launcher requires Caddy to be installed, configured, and active before
`start`. A fresh Ubuntu image can be bootstrapped with:

```sh
sudo apt-get update
sudo apt-get install -y caddy
sudoedit /etc/caddy/Caddyfile
```

A minimal local configuration for the initial preflight is:

```caddyfile
:8080 {
    reverse_proxy 127.0.0.1:8090
}
```

Validate and activate it:

```sh
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl enable --now caddy
systemctl is-active caddy
```

Replace the bootstrap listener with the production TLS/authentication route as
appropriate. It should proxy to `127.0.0.1:8090` with streaming flush enabled
and a response timeout long enough for initial model loading. Keep port 8090
loopback-only; do not bypass Caddy's authentication at the firewall.

## Agent-client completion limits

Set an explicit `max_tokens` or `max_completion_tokens` in agent clients. A
4,096–8,192 token limit is a practical starting point. The server repairs a
simple unterminated DSML tool call when generation reaches the request limit;
a 32,000-token client limit can therefore spend tens of minutes decoding a
malformed tool argument before the repair runs. This is a model-output/API
failure mode, not evidence that RDMA stopped making progress. Health checks
remain responsive after the affected request completes.
