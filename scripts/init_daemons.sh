#!/usr/bin/env bash
# Manage init_daemon across capture1-6.
#
# Usage:
#     bash scripts/init_daemons.sh start
#     bash scripts/init_daemons.sh start --p2-semantic
#     bash scripts/init_daemons.sh stop
#     bash scripts/init_daemons.sh status
#     bash scripts/init_daemons.sh log capture1   # tail one PC's log
set -euo pipefail

PCS=(capture1 capture2 capture3 capture5 capture6)  # capture4 out
# Resolved on the REMOTE side via $HOME — do NOT expand ~ locally.
PY='$HOME/anaconda3/envs/gotrack_cu128/bin/python'
DAEMON='$HOME/AutoDex/src/execution/daemon/init_daemon.py'
LOG=/tmp/init_daemon.log

ACTION="${1:-status}"
P2_SEMANTIC_ARGS=""
if [[ "$ACTION" == "start" ]]; then
    P2_SEMANTIC_FLAG="${2:-}"
    if [[ "$P2_SEMANTIC_FLAG" == "--p2-semantic" ]]; then
        P2_SEMANTIC_ARGS="--port-semantic 5010"
    elif [[ -n "$P2_SEMANTIC_FLAG" ]]; then
        echo "usage: $0 {start [--p2-semantic]|stop|status|log [pc_name]}"
        exit 1
    fi
fi

case "$ACTION" in
    start)
        for pc in "${PCS[@]}"; do
            ssh -o ConnectTimeout=3 "$pc" "pkill -9 -f init_daemon 2>/dev/null || true" &
        done
        wait
        sleep 2
        for pc in "${PCS[@]}"; do
            ssh -o ConnectTimeout=3 "$pc" "bash -c 'nohup $PY $DAEMON $P2_SEMANTIC_ARGS > $LOG 2>&1 &'"
        done
        sleep 3
        for pc in "${PCS[@]}"; do
            n=$(ssh -o ConnectTimeout=3 "$pc" "pgrep -fc 'python.*init_daemon'" 2>/dev/null || echo 0)
            echo "  $pc: $n daemon(s)"
        done
        ;;
    stop)
        for pc in "${PCS[@]}"; do
            ssh -o ConnectTimeout=3 "$pc" "pkill -9 -f init_daemon 2>/dev/null && echo killed || true" &
        done
        wait
        ;;
    status)
        for pc in "${PCS[@]}"; do
            # ``pgrep -f`` can count the remote shell because the pattern is
            # present in its own command string. The bracketed grep expression
            # avoids self-matching; ``|| true`` preserves a meaningful 0.
            n=$(ssh -o ConnectTimeout=3 "$pc" "ps -eo args | grep -c '[p]ython .*init_daemon.py' || true" \
                2>/dev/null || echo "?")
            echo "  $pc: $n"
        done
        ;;
    log)
        pc="${2:-capture1}"
        ssh "$pc" "tail -50 $LOG"
        ;;
    *)
        echo "usage: $0 {start [--p2-semantic]|stop|status|log [pc_name]}"
        exit 1
        ;;
esac
