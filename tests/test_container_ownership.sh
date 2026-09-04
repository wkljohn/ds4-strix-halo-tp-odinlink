#!/bin/bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
launcher=$repo/deploy/ds4-tp-caddy.sh
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/artifacts" "$tmp/runtime" "$tmp/research"

owned=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
other=abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789
log=$tmp/calls.log

cat >"$tmp/config" <<EOF
MODEL=$tmp/model.gguf
PEER_MGMT=peer
PEER_HOST_KEY_ALIAS=peer
PEER_REPO=$repo
COORDINATOR_RDMA_ADDR=192.0.2.1
RDMA_PROFILE=roce-v2
DS4_TP_CONTAINER=1
DS4_TP_CONTAINER_IMAGE=test/image:fixed
DS4_TP_CONTAINER_IMAGE_ID=$owned
DS4_TP_CONTAINER_VOLUME=$tmp/artifacts
DS4_TP_CONTAINER_WRITABLE_VOLUME=$tmp/runtime
DS4_RESEARCH_ROOT=$tmp/research
DS4_PEER_RESEARCH_ROOT=$tmp/research
EOF

cat >"$tmp/bin/systemctl" <<'EOF'
#!/bin/bash
exit 1
EOF
cat >"$tmp/bin/podman" <<'EOF'
#!/bin/bash
set -eu
printf 'local %s\n' "$*" >>"$TEST_CALL_LOG"
if [[ $1 == inspect ]]; then
  printf '%s\n' "$TEST_LOCAL_IMAGE"
fi
EOF
cat >"$tmp/bin/ssh" <<'EOF'
#!/bin/bash
set -eu
cmd=${!#}
printf 'peer %s\n' "$cmd" >>"$TEST_CALL_LOG"
if [[ $cmd == *"podman inspect"* ]]; then
  printf '%s\n' "$TEST_PEER_IMAGE"
fi
EOF
chmod +x "$tmp/bin/systemctl" "$tmp/bin/podman" "$tmp/bin/ssh"

run_stop() {
  : >"$log"
  PATH="$tmp/bin:$PATH" DS4_DEPLOY_CONFIG="$tmp/config" \
    TEST_CALL_LOG="$log" TEST_LOCAL_IMAGE=$1 TEST_PEER_IMAGE=$2 \
    "$launcher" stop
}

if run_stop "$other" "$owned" >"$tmp/out" 2>"$tmp/err"; then
  echo 'FAIL mismatched local image was accepted' >&2
  exit 1
fi
grep -q 'coordinator name/unit is not owned' "$tmp/err"
! grep -q 'podman stop' "$log"

if run_stop "$owned" "$other" >"$tmp/out" 2>"$tmp/err"; then
  echo 'FAIL mismatched peer image was accepted' >&2
  exit 1
fi
grep -q 'peer container name .* is not owned' "$tmp/err"
! grep -q 'podman stop' "$log"

run_stop "$owned" "$owned" >"$tmp/out" 2>"$tmp/err"
grep -q 'local stop -t 60 ds4-tp-coordinator' "$log"
grep -q "peer podman stop -t 60 'ds4-tp-worker'" "$log"

echo 'PASS container-ownership'
