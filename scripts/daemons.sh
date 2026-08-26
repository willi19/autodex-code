#!/usr/bin/env bash
# Drive every capture-PC daemon the current pipeline needs, in one command.
#
# run_auto needs BOTH init_daemon (mask+pose) and snapshot_daemon, and they are
# started/stopped as a set — a half-restarted pair is the kind of state that
# shows up much later as "0/20 masks" rather than as an error.
#
# Usage:
#     bash scripts/daemons.sh start          # restart both, everywhere
#     bash scripts/daemons.sh stop
#     bash scripts/daemons.sh status
#     bash scripts/daemons.sh log capture1   # tail both logs on one PC
#
# Narrow it to one kind with a second word:
#     bash scripts/daemons.sh start init
#     bash scripts/daemons.sh status snapshot
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-status}"
WHICH="${2:-}"

# `log` takes a PC name in $2, not a daemon kind.
if [[ "$ACTION" == "log" ]]; then
    PC="${2:-capture1}"
    for kind in init snapshot; do
        echo "===== ${kind}_daemon @ ${PC} ====="
        bash "$HERE/${kind}_daemons.sh" log "$PC"
    done
    exit 0
fi

case "$WHICH" in
    init)     KINDS=(init) ;;
    snapshot) KINDS=(snapshot) ;;
    ""|all)   KINDS=(init snapshot) ;;
    *) echo "unknown daemon kind: $WHICH (use init | snapshot | all)"; exit 2 ;;
esac

rc=0
for kind in "${KINDS[@]}"; do
    echo "===== ${kind}_daemon: ${ACTION} ====="
    bash "$HERE/${kind}_daemons.sh" "$ACTION" || rc=$?
done

# `start`/`status` print one line per PC per daemon; a PC missing from either
# list is the failure worth seeing, so summarise rather than make the caller
# eyeball two blocks.
if [[ "$ACTION" == "start" || "$ACTION" == "status" ]]; then
    echo "===== summary ====="
    for kind in "${KINDS[@]}"; do
        n=$(bash "$HERE/${kind}_daemons.sh" status 2>/dev/null | grep -c ': 1')
        total=$(bash "$HERE/${kind}_daemons.sh" status 2>/dev/null | grep -c ':')
        echo "  ${kind}_daemon: ${n}/${total} PCs up"
        [[ "$n" == "$total" ]] || rc=1
    done
fi

exit $rc
