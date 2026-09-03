# DS4 Strix Halo TP over OdinLink, RoCE v2, or InfiniBand

Tensor-parallel **DeepSeek V4 Flash 0731** inference across two AMD Strix Halo
APUs. This fork runs the 153.32 GiB Q4_K model over consumer USB4/TB5 with
OdinLink GPU RDMA, over a Mellanox RoCE v2 link, or over a directly connected
native Mellanox InfiniBand cable (ConnectX-3, `mlx4`).

- **2 × Ryzen AI MAX+ 395 / Radeon 8060S**
- **2 × 128 GB installed RAM; 96 GiB ROCm aperture per node**
- **Tensor parallelism = 2**
- **Cache-free Q4_K production defaults**
- **256K-context server profile**

```text
 Ryzen AI MAX+ 395          RDMA link          Ryzen AI MAX+ 395
   Radeon 8060S       <================>          Radeon 8060S
      TP rank 0    OdinLink, RoCE v2, or IB          TP rank 1
```

## Performance (DeepSeek V4 Flash 0731)

| DeepSeek V4 0731 TP=2 configuration | Measurement | Prefill | Decode | Status |
|---|---|---:|---:|---|
| Original Q4_K baseline | archived pre-acceleration TP=2 run | **34.11 t/s** | **9.96 t/s** | historical baseline, not single-node scaling |
| **Huihui Q2_K over RoCE v2** | balanced 50/50, 2,048-token chunk | **209.60 t/s** | **20.09 t/s** | recorded candidate control; rollback 213.50/20.07, FNV unchanged |
| **Antirez Q4_K over OdinLink** | balanced 50/50, 2,048-token chunk | **233.04 t/s** | **19.17 t/s** | three-run median, exact fingerprint |
| **Antirez Q4_K over RoCE v2** | balanced 50/50, 2,048-token chunk | **280.58 t/s** | **21.30 t/s** | recorded candidate control; rollback 276.59/21.24, FNV unchanged |
| **Current Q4_K + DSpark** | 46/54 split | — | — | experimental revalidation pending |

### GLM-5.3 Flash TP=2 research-branch performance

These `ds4-bench-tp` results use the diverse 4,096-token prefill plus
300-token decode workload at batch 256. They require research branch
[`research/glm5-kb-lds-exact`](https://github.com/wkljohn/ds4-strix-halo-tp-odinlink/tree/ee9ca905f090a974b56fbfc2b92ed3d821dd0dd6)
at commit `ee9ca90`; they are not yet enabled by `main`.

| Configuration | Measurement | Prefill | Decode | Status |
|---|---|---:|---:|---|
| **GLM-5.3 Flash Q4_K over RoCE v2** | three source-clean runs | **78.59 t/s** | **9.45 t/s** | three-run median; runs were 78.59/9.45, 77.78/9.54, and 79.05/9.42, FNV `9012bd4d7c5ce422` |
| **GLM-5.3 Flash Q4_K over OdinLink** | one matched provider run | **76.00 t/s** | **9.43 t/s** | zero fallback; FNV `9012bd4d7c5ce422` |
| **GLM-5.3 Flash Q2 over RoCE v2** | 2,048-token control workload | **21.87–21.95 t/s** | **3.48–3.49 t/s** | opt-in mixed IQ2_XXS/Q2_K path; repeated fingerprint matched |

The Q4_K candidate keeps all 4,096 prompt rows batched, adds no persistent
weight cache, and uses 45.07 MiB of reusable scratch per rank. It remains
default-off pending its Lane-B quality review and three-run OdinLink median.

The DeepSeek table above uses `ds4-bench-tp`: a fixed 2,048-token prefill
followed by 300 generated tokens over mandatory RDMA. Its current Q4_K rows
use the Antirez reference model listed below. It does not fit one node's
current 96 GiB ROCm aperture; TP=2 keeps one expert shard on each node. Q2_K
and Q4_K run without a persistent expanded-weight cache.

The `main` branch tracks the pinned ROCm 7.14 gfx1151 toolchain used for these
release results.

GLM-5.3 Flash support is an experimental TP=2 staged path in this branch; it
keeps the model's KDA state sharded by attention head and does not change the
validated DeepSeek production path. The GLM row is not an ordinary server
benchmark: full session integration remains fail-closed until promoted.

The ordinary benchmark and deployment launchers enable the validated ordered
ROCm TP callback, temporal-compressor schedule, shape-gated M256/K128 Q8
projection, cooperative HC decode stage, exact long-context indexer top-k, and
RoCE prefill wavefront automatically. The wavefront provider-gates itself off
on OdinLink. DSpark stays opt-in and does not inherit that target-only
schedule.

The table reports reproducible inference results, not a single-node scaling
claim. Raw runs, fingerprints, kernel decisions, rejected candidates, memory
policy, and maintainer gates are preserved locally under
`$DS4_RESEARCH_ROOT/reports/strix-halo-tp-validation-2026-08/`.

## Supported and tested models

| Model source | Tested target files | Support |
|---|---|---|
| [Antirez DeepSeek V4 GGUF](https://huggingface.co/antirez/deepseek-v4-gguf) | `DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf` (164,633,502,592 bytes) | **Recommended.** Used for the current Q4_K OdinLink and RoCE v2 rows. |
| [Huihui DeepSeek V4 Flash 0731 GGUF](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF) | `DeepSeek-V4-Flash-Q2_K-0731.gguf`; `DeepSeek-V4-Flash-Q4_K-0731.gguf` | Supported. The Q2_K file is used for the current Q2_K row. |
| [GLM-5.3 Flash GGUF](https://huggingface.co/antirez/glm-5.3-flash-gguf) | GLM-5.3 Flash Q4/Q2 GGUF targets | **Experimental only.** TP=2 staged support; not yet an ordinary production server path. |
| Unsloth DeepSeek V4 Flash 0731 `UD-*` target weights | — | **Not supported:** their mixed-precision tensor layouts do not match the currently validated DS4 target paths. |

The Unsloth warning applies to `UD-*` target-model weights, not to a separately
documented optional DSpark drafter.

## Quick start

Both nodes need ROCm support for `gfx1151`, passwordless SSH from the
coordinator to the worker, and their own local copy of the same repository
commit and GGUF at the same absolute paths. Their filesystems are not shared.

Follow [DS4 on Strix Halo](STRIXHALO.md) for the Ubuntu 26.04 ROCm 7.14
tarball, rocWMMA header isolation, memory-layout decision, Secure Boot check,
and coordinator-to-worker SSH setup. Ubuntu 26.04 apt currently supplies ROCm
7.1; it is not a substitute for the validated 7.14 bundle.

Build on both nodes:

```sh
git clone https://github.com/wkljohn/ds4-strix-halo-tp-odinlink.git
cd ds4-strix-halo-tp-odinlink
HIP_PATH=/absolute/path/to/rocm-7.14.0 \
CPATH=/absolute/path/to/rocm-7.14.0/include \
CPLUS_INCLUDE_PATH=/absolute/path/to/rocm-7.14.0/include \
DS4_ROCM_HOME=/absolute/path/to/rocm-7.14.0 \
  make -j"$(nproc)" strix-halo
```

`DS4_ROCM_HOME` must name the ROCm 7.14 installation root containing
`bin/hipcc`; setting it explicitly prevents an older `/opt/rocm` installation
from being selected accidentally.

Create the benchmark configuration on node 1:

```sh
cp bench.env.example bench.env.local
$EDITOR bench.env.local
```

Set the peer SSH address, peer repository path, coordinator RDMA address,
transport devices, output directory, and—when using OdinLink—the local
`OdinLink-Five` path. The launcher then verifies both binaries, samples both
model copies, starts both cold model loads together, requires explicit RDMA,
and rejects transport fallback.

Run one fixed Q4_K benchmark:

```sh
./run-tp-ds4-bench.sh q4-r1 \
  /absolute/path/DeepSeek-V4-Flash-Q4_K-0731.gguf
```

Select `DS4_BENCH_RDMA_PROFILE=roce-v2` in `bench.env.local` for Mellanox
RoCE v2, or `ib-mlx4` for native Mellanox InfiniBand.
Use distinct tags (`q4-r1`, `q4-r2`, `q4-r3`) and report the median. Candidate
fingerprints and the combined pre-main test belong to the
local validation archive at
`$DS4_RESEARCH_ROOT/reports/strix-halo-tp-validation-2026-08/`, not
normal installation.

## OdinLink over USB4/TB5

Install the OdinLink driver on both hosts using the
[OdinLink-Five instructions](https://github.com/wkljohn/OdinLink-Five). The
validated userspace provider is pinned below:

```sh
sudo apt install build-essential cmake linux-headers-"$(uname -r)" \
  libibverbs-dev rdma-core pkg-config libglib2.0-dev
git clone https://github.com/wkljohn/OdinLink-Five.git
git -C OdinLink-Five checkout 8a77ccbf051b5a615a2b4d9a75ede10af524614a
cmake -S OdinLink-Five -B OdinLink-Five/build \
  -DBUILD_VERBS=ON -DBUILD_DAEMON=ON -DBUILD_TRAY=OFF
cmake --build OdinLink-Five/build -j"$(nproc)" \
  --target driver odl_tb5_cli odl_tb5_verbs odl_tb5_verbs_provider
```

Disable Secure Boot or enroll a trusted module-signing key before loading the
out-of-tree driver. Load it on both nodes; the character device is created only
after both peers advertise the OdinLink Thunderbolt service. Then verify the
device and link before loading the model:

```sh
ls -l /dev/odl_tb5_0
# Node 1
./OdinLink-Five/build/cli/odl_tb5_cli --server --device 0
# Node 2
./OdinLink-Five/build/cli/odl_tb5_cli --client --device 0 --test latency
```

Set `DS4_ODINLINK_ROOT`, both `odl_tb5_0` device names, and the coordinator's
OdinLink address in `bench.env.local`. The optimized provider and Strix Halo
settings are already defaults; users should not copy diagnostic environment
switches into production.

## Mellanox RoCE v2

Install system verbs on both nodes and keep OdinLink provider variables out of
the RoCE environment:

```sh
sudo apt install rdma-core ibverbs-utils perftest libibverbs-dev
unset DS4_TP_VERBS_LIB ODL_VERBS_WC_STREAM_COPY DS4_TP_ODINLINK_BATCH_ASYNC
```

Give the directly connected interfaces addresses on one subnet and enable a
9,000-byte MTU. Substitute each host's netdev and address:

```sh
sudo ip link set <mlx5-netdev> mtu 9000 up
sudo ip addr replace <node-address>/24 dev <mlx5-netdev>
```

Find an IPv4-mapped RoCE v2 GID and prove that mlx5 can register DS4's mapped
communication slab:

```sh
cat /sys/class/infiniband/<mlx5-device>/ports/1/gid_attrs/types/<gid-index>
make tests/roce_v2_mr_probe
./tests/roce_v2_mr_probe <mlx5-device>
```

The GID output must be `RoCE v2`; the probe must pass its mapped-host,
three-MR layout. Put the coordinator address, both device names, and GID index
in `bench.env.local`, set `DS4_BENCH_RDMA_PROFILE=roce-v2`, and run the same
benchmark command. Reference hardware details and the registration pitfall are
documented locally under `$DS4_RESEARCH_ROOT/reports/roce-v2-2026-08-14/`.

## Mellanox InfiniBand (ConnectX-3, mlx4)

A directly connected native InfiniBand cable (no RoCE GID-type or 9,000-byte
MTU configuration) uses the generic verbs path: RC queue pairs, 16 KiB decode
framing, and a single full-slab MR. RC is required because the exchanges are
full-duplex post-recv/post-send: a UC SEND that reaches the peer before its
RECV WR is armed is dropped silently (no RNR retry) and hangs the round,
while RC retransmits on RNR. Install the same system verbs on both nodes and
keep the OdinLink provider variables unset:

```sh
sudo apt install rdma-core ibverbs-utils opensm
unset DS4_TP_VERBS_LIB ODL_VERBS_WC_STREAM_COPY DS4_TP_ODINLINK_BATCH_ASYNC
```

Run a subnet manager on one node so both directly connected ports receive
LIDs, and give the InfiniBand network devices addresses on one subnet for the
TCP control plane:

```sh
sudo opensm
sudo ip addr replace <node-address>/24 dev <ib-netdev>
rdma link show    # both ports must be ACTIVE with LIDs 1 and 2
```

Prove that the device can register DS4's mapped communication slab. GID 0 is
the canonical native-IB GID and is also selected automatically when no
IPv4-mapped GID exists:

```sh
ibv_devinfo -d <ib-device>
make tests/roce_v2_mr_probe
./tests/roce_v2_mr_probe <ib-device>
```

In `bench.env.local`, set `DS4_BENCH_RDMA_PROFILE=ib-mlx4`, the coordinator's
InfiniBand interface address, both device names from `ibv_devinfo -l`
(udev may name them like `ibp195s0`), and `DS4_RDMA_GID_INDEX=0`; then run the
same benchmark command. For a manual two-node run, pass
`--transport rdma --rdma-device <ib-device> --rdma-gid-index 0` on both ranks.

## Production server

The included 256K profile starts both independent model loads together, keeps
the API on `127.0.0.1:8090`, and supports either RDMA transport behind Caddy:

```sh
cp deploy/config.env.example deploy/config.env.local
sed -i "s/^DS4_SERVER_SHA256=.*/DS4_SERVER_SHA256=$(sha256sum ./ds4-server | awk '{print $1}')/" \
  deploy/config.env.local
$EDITOR deploy/config.env.local
deploy/ds4-tp-caddy.sh start
deploy/ds4-tp-caddy.sh status
```

See [deploy/README.md](deploy/README.md) for the required Caddy route, RoCE
memlock handling, logs, and stop/restart commands.

OpenCode can use the deployed DeepSeek service as an OpenAI-compatible provider
by pointing its base URL at `http://127.0.0.1:8090/v1` (or the Caddy URL),
selecting the model name returned by `GET /v1/models`, and supplying the server
API key if authentication is enabled. GLM-5.3 is not selectable from this
server yet: its staged TP harness is available for development, while ordinary
GLM5.3 sessions intentionally fail closed until full session integration and
an end-to-end correctness gate are promoted.

Agent clients should send an explicit completion limit. A practical starting
point is 4,096–8,192 tokens: if the model leaves a DSML tool call open, DS4
repairs it at the request limit, so an unnecessarily large limit can look like
a stalled request even while decode is progressing.

## Research and maintenance

- Local TP validation archive: `$DS4_RESEARCH_ROOT/reports/strix-halo-tp-validation-2026-08/`
- Local RoCE v2 A/B archive: `$DS4_RESEARCH_ROOT/reports/roce-v2-2026-08-14/`
- [OdinLink integration record](ODINLINK.md)
- [Kernel and architecture reports](docs/)
- [OpenAI-compatible server benchmark](docs/API-BENCHMARK.md)
- [Safe upstream synchronization](docs/UPSTREAM-SYNC.md)

To prepare—but not automatically commit—an upstream update:

```sh
git switch main
git pull --ff-only origin main
./scripts/prepare-upstream-sync.sh
```

## When to use upstream DS4

This fork is for **dual-Strix-Halo TP=2 over OdinLink, RoCE v2, or
InfiniBand**. For general
DS4 usage—including model downloads, single-node inference, Metal, CUDA,
multi-GPU CUDA, SSD streaming, pipeline parallelism, the server, and the coding
agent—use the canonical [antirez/ds4](https://github.com/antirez/ds4) README and
[DwarfStar documentation](https://dwarfstar.sh/docs/quickstart/).

The fork preserves upstream backends and non-TP paths, but does not duplicate
their user manual. Canonical changes are brought in through the documented
review branch so the Strix Halo and RDMA paths can be revalidated before merge.

## Acknowledgements

This project is a fork of [DS4](https://github.com/antirez/ds4) and retains its
license and attribution. DS4 in turn depends on ideas and formats pioneered by
[llama.cpp](https://github.com/ggml-org/llama.cpp) and GGML. See
[LICENSE](LICENSE) and the repository history for complete notices.
