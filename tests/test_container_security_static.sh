#!/bin/bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
launcher=$repo/deploy/ds4-tp-caddy.sh
config=$repo/deploy/config.env.example
containerfile=$repo/deploy/Containerfile.rocm714-rdma

grep -q 'DS4_TP_CONTAINER_IMAGE_ID' "$launcher"
grep -q 'DS4_TP_CONTAINER_IMAGE_ID must be the 64-hex content ID' "$launcher"
grep -q 'peer container image ID differs' "$launcher"
grep -q -- '--entrypoint /bin/bash' "$launcher"
grep -q -- '--cap-drop=all' "$launcher"
grep -q -- '--security-opt no-new-privileges' "$launcher"
grep -q -- '--group-add keep-groups' "$launcher"
grep -q -- ':ro' "$launcher"
grep -q '^FROM .*@sha256:' "$containerfile"
grep -q 'ibverbs-providers=' "$containerfile"
grep -q 'ibverbs-utils=' "$containerfile"
grep -q 'command -v ibv_devinfo' "$launcher"

if grep -q -- '--security-opt label=disable' "$launcher"; then
  echo 'FAIL label separation is disabled by default' >&2
  exit 1
fi
if grep -q -- '--security-opt unmask=all' "$launcher"; then
  echo 'FAIL all masked paths are unmasked by default' >&2
  exit 1
fi
if grep -q 'src=/usr/lib/.*/libibverbs' "$launcher"; then
  echo 'FAIL host libibverbs providers are mounted into the container' >&2
  exit 1
fi
grep -q 'same content-addressed image ID' "$config"
if grep -q 'DS4_TP_CONTAINER_VOLUME=/home' "$config"; then
  echo 'FAIL example mounts a home directory' >&2
  exit 1
fi
echo 'PASS container-security-static'
