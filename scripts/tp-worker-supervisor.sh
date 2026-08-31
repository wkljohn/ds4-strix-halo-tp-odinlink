#!/bin/bash
# Run one TP worker and persist its terminal status for the invoking launcher.
# The status file is diagnostic evidence; this wrapper does not alter worker
# arguments or environment.  It forwards termination so cleanup remains safe.
set -u

status_file=${1:?status file required}
shift
child_pid=
write_status() {
    rc=$1
    if (( rc >= 128 )); then
        signal=$((rc - 128))
    else
        signal=0
    fi
    printf 'exit_code=%d\nsignal=%d\n' "$rc" "$signal" > "$status_file"
}
forward_term() {
    if [[ -n ${child_pid:-} ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
    fi
}
trap forward_term TERM INT

"$@" &
child_pid=$!
wait "$child_pid"
rc=$?
child_pid=
write_status "$rc"
exit "$rc"
