# DS4 on Strix Halo

This is the bare-metal setup for DS4 ROCm inference on a Strix Halo machine
with 128 GiB of physical memory and a Radeon 8060S (`gfx1151`). Apply the same
ROCm and build setup on both tensor-parallel nodes.

## 1. Install one ROCm toolchain

Ubuntu 26.04's ROCm packages provide ROCm 7.1, not the ROCm 7.14 toolchain
validated by this branch. Install the gfx1151 TheRock bundle separately. The
bundle used for the current deployment is:

```text
https://rocm.prereleases.amd.com/tarball-multi-arch/therock-dist-linux-gfx1151-7.14.0rc3.tar.gz
```

Extract it as `/opt/rocm-7.14.0` and verify the resulting root before building:

```sh
test -x /opt/rocm-7.14.0/bin/hipcc
/opt/rocm-7.14.0/bin/hipcc --version
```

Install the host-side build dependencies, but do **not** install Ubuntu's
`librocwmma-dev` alongside the bundle:

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config \
  linux-headers-"$(uname -r)" libibverbs-dev rdma-core libglib2.0-dev mokutil
sudo apt-get remove -y librocwmma-dev
```

`hipcc` can otherwise find `/usr/include/rocwmma/rocwmma.hpp` before the
bundle's matching headers. Ubuntu's header tree has no `rocwmma/internal/`, so
the build then fails at `internal/accessors.hpp`. Do not combine Ubuntu's
rocWMMA headers, a manually copied `/usr/local/include/rocwmma`, and TheRock.
Removing the foreign header packages/copies is the preferred repair. If a
managed machine cannot remove them, its build wrapper must put
`-isystem /opt/rocm-7.14.0/include` before all system include paths.

Do not create `/opt/rocm` symlinks to `/usr`. Select the complete installation
explicitly:

```sh
HIP_PATH=/opt/rocm-7.14.0 \
CPATH=/opt/rocm-7.14.0/include \
CPLUS_INCLUDE_PATH=/opt/rocm-7.14.0/include \
DS4_ROCM_HOME=/opt/rocm-7.14.0 \
  make -j"$(nproc)" strix-halo
```

These include exports are the tested fresh-install recipe and keep the TheRock
header tree ahead of `/usr/include`. `make rocm` is an alias for
`make strix-halo`. Check the compiler printed by the build if a host has ever
had more than one ROCm installation.

## 2. Enable ROCm access

The user running DS4 must be able to open `/dev/kfd` and the DRM render node:

```sh
sudo usermod -aG render,video "$USER"
```

Log out and back in, or reboot. Verify:

```sh
/opt/rocm-7.14.0/bin/rocminfo | grep -A80 'Name:                    gfx1151'
```

If DS4 says `no ROCm-capable device is detected`, check that `rocminfo` can
open `/dev/kfd` and that `groups` includes `render`.

## 3. Classify memory before changing GTT

Strix Halo firmware can expose memory in two materially different ways. Check
the GPU-visible VRAM and remaining operating-system RAM first:

```sh
for f in /sys/class/drm/card*/device/mem_info_vram_total; do
  printf '%s: ' "$f"
  numfmt --to=iec < "$f"
done
free -h
cat /proc/cmdline
```

If `mem_info_vram_total` is already about **96 GiB** and the OS has only about
**30 GiB**, make no GTT change. Adding `amdgpu.gttsize=126976` on that firmware
policy oversubscribes the same 128 GiB of physical memory.

The large-GTT recipe below is only for the opposite firmware policy: a small
VRAM carve-out with most RAM left to the OS. On such a host, and only after
confirming that layout, the known recipe is:

```text
amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856
```

After changing GRUB and rebooting, verify `/proc/cmdline`, the VRAM/GTT kernel
messages, `rocminfo`, and `free -h` again. Never copy this stanza between
machines solely because they have the same processor or total RAM.

The 96 GiB field setup used `amd_iommu=off`, but the observation did not prove
that OdinLink requires it. Treat IOMMU policy as a separate platform decision;
do not infer it from the VRAM/GTT classification alone.

## 4. Prepare OdinLink on both nodes

Check Secure Boot before building or loading the out-of-tree driver:

```sh
mokutil --sb-state
cat /sys/module/module/parameters/sig_enforce 2>/dev/null || true
```

With Secure Boot/EFI lockdown enforcing signatures, an unsigned
`odl_tb5.ko` fails with `Key was rejected by service`. This cannot be repaired
over SSH by rebuilding or retrying `insmod`: disable Secure Boot in UEFI
firmware (or establish a signed-module/MOK workflow), reboot, and check again.

OdinLink's daemon remains enabled when the tray is disabled, so
`libglib2.0-dev` is required even with `-DBUILD_TRAY=OFF`:

```sh
git clone https://github.com/wkljohn/OdinLink-Five.git
git -C OdinLink-Five checkout d0b54fc6e6adb20cf88926ca0cf60eed51527b31
cmake -S OdinLink-Five -B OdinLink-Five/build \
  -DBUILD_VERBS=ON -DBUILD_DAEMON=ON -DBUILD_TRAY=OFF
cmake --build OdinLink-Five/build -j"$(nproc)" \
  --target driver odl_tb5_cli odl_tb5_verbs odl_tb5_verbs_provider
```

`libfuse3-dev` and `libssl-dev` enable optional daemon file-access and sync
features; DS4's verbs transport does not require them.

Load the built driver on **both** nodes. Loading one side is not sufficient:
`/dev/odl_tb5_0` appears only after both Thunderbolt peers advertise the
`tbsvc:kodinlink...` service and bind successfully. Confirm both sides before
starting DS4:

```sh
sudo insmod OdinLink-Five/driver/odl_tb5.ko
sudo dmesg | grep -E 'tbsvc:kodinlink|odl_tb5'
ls -l /dev/odl_tb5_0
```

A `thunderbolt-net` interface or IP address on `thunderbolt0` is not an
OdinLink device and does not satisfy this check. Do not start a one-sided DS4
run until `/dev/odl_tb5_0` exists on both ranks.

## 5. Preserve the management network

Keep Ubuntu Desktop's existing NetworkManager ownership of the management
interface. Do not add a netplan file with `renderer: networkd` for the whole
host: it can take the Ethernet interface away from NetworkManager, renew DHCP,
and change the address used for SSH.

Use the ordinary Ethernet/Wi-Fi management address for SSH. Keep it separate
from any Thunderbolt or RDMA data-path address, and do not depend on the
OdinLink link for recovery access.

Create a NetworkManager profile for the Thunderbolt data address instead of a
host-wide `networkd` netplan file. Use `.1` on the coordinator and `.2` on the
worker:

```sh
sudo nmcli connection add type ethernet ifname thunderbolt0 \
  con-name odinlink-tb ipv4.method manual ipv4.addresses 10.4.0.1/24 \
  ipv4.never-default yes ipv6.method disabled
sudo nmcli connection up odinlink-tb
```

This profile must not change the NetworkManager connection for the management
interface. Set `COORDINATOR_RDMA_ADDR` to the coordinator's Thunderbolt address.

The deployment launcher SSHes **from the coordinator to the worker**. Create
and install the key in that direction, then prove the exact configured alias
works non-interactively:

```sh
# Run on the coordinator.
ssh-keygen -t ed25519
ssh-copy-id user@worker-management-address
ssh -o HostKeyAlias=ds4-worker-management \
  user@worker-management-address true
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o HostKeyAlias=ds4-worker-management \
  user@worker-management-address true
```

Set `PEER_MGMT` to that management destination and `PEER_HOST_KEY_ALIAS` to
the same unique alias used above. Do not reuse a Thunderbolt IP that identifies
another fabric. The first SSH command using `HostKeyAlias` records the trusted
key under the exact name the launcher's `StrictHostKeyChecking=yes` lookup
will use.

## 6. Verify artifacts and run

The nodes have independent filesystems. Finish copying the GGUF before launch,
then verify the repository commit, executable hashes, model size, and sampled
model fingerprint on both sides. The benchmark and deployment launchers
perform these checks and refuse transport fallback.

For OdinLink, also run the standalone server/client test documented in the
main [README](README.md#odinlink-over-usb4tb5) before loading the model. Use the
standard Q4_K model listed there; mixed target-model layouts that are not in
the compatibility table remain unsupported.
