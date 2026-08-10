#!/usr/bin/env bash
# Verify that OdinLink recovery can be observed through an independent network.
#
# This tool is deliberately check-only.  Live module reload can make the peer
# tear down streams while Thunderbolt completions are queued, which has caused
# a kernel panic.  Never point the management host at 10.4.0.2.
set -euo pipefail

if [[ ${1:-} == "--reload" ]]; then
    cat >&2 <<'EOF'
REFUSED: live OdinLink module reload is disabled.

A peer logout can race a queued Thunderbolt TX completion in affected drivers
and panic the other node. Use a controlled reboot on both nodes until the
driver lifetime fix has been built, deployed, and validated.

Do not unbind the Thunderbolt PCI controller as a substitute: controller
unbind can block inside the kernel and make the node unmanageable.
EOF
    exit 1
elif [[ ${1:-} == "--check" ]]; then
    shift
fi
if (($# != 0)); then
    echo "usage: ODL_PEER_MGMT_HOST=<independent-host> $0 [--check]" >&2
    exit 2
fi

PEER_HOST=${ODL_PEER_MGMT_HOST:-}
LOCAL_ROOT=${ODL_DRIVER_ROOT:-/home/wkljohn/Desktop/cc/OdinLink-Five}
PEER_ROOT=${ODL_PEER_DRIVER_ROOT:-/home/wkljohn/Desktop/cc/OdinLink-Five}
SSH_USER=${ODL_PEER_MGMT_USER:-wkljohn}

if [[ -z $PEER_HOST ]]; then
    echo "error: ODL_PEER_MGMT_HOST is required; the OdinLink data address is not a management path" >&2
    exit 1
fi

PEER_IP=$(getent ahostsv4 "$PEER_HOST" | awk 'NR == 1 { print $1 }')
if [[ -z $PEER_IP ]]; then
    echo "error: cannot resolve independent peer management host '$PEER_HOST'" >&2
    exit 1
fi
if [[ $PEER_IP == 10.4.* ]]; then
    echo "error: refusing peer management address $PEER_IP on the OdinLink data subnet" >&2
    exit 1
fi

ROUTE=$(ip route get "$PEER_IP")
ROUTE_DEV=$(awk '{ for (i=1; i<=NF; i++) if ($i == "dev") { print $(i+1); exit } }' <<<"$ROUTE")
case "$ROUTE_DEV" in
    bond*|thunderbolt*|odl*|"")
        echo "error: peer management route uses unsafe interface '$ROUTE_DEV': $ROUTE" >&2
        exit 1
        ;;
esac

PEER=${SSH_USER}@${PEER_HOST}
REMOTE_CHECK='set -eu
client_ip=${SSH_CONNECTION%% *}
route=$(ip route get "$client_ip")
dev=$(printf "%s\n" "$route" | awk '\''{ for (i=1; i<=NF; i++) if ($i == "dev") { print $(i+1); exit } }'\'')
case "$dev" in bond*|thunderbolt*|odl*|"") echo "unsafe return route: $route" >&2; exit 1;; esac
printf "peer management return route: %s\n" "$route"
sudo -n true
test -f '"$PEER_ROOT"'/driver/odl_tb5.ko
test "$(uname -r)" = "$(modinfo -F vermagic '"$PEER_ROOT"'/driver/odl_tb5.ko | awk '\''{print $1}'\'')"'

echo "local management route: $ROUTE"
ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" "$REMOTE_CHECK"
sudo -n true
test -f "$LOCAL_ROOT/driver/odl_tb5.ko"
if [[ $(modinfo -F vermagic "$LOCAL_ROOT/driver/odl_tb5.ko" | awk '{print $1}') != $(uname -r) ]]; then
    echo "error: local OdinLink module does not match running kernel" >&2
    exit 1
fi

echo "SAFE_CHECK_OK: independent management paths verified; no module state changed"
