#!/usr/bin/env bash
# Manage gotrack_daemon across capture1-6.
#
# Usage:
#     bash scripts/gotrack_daemons.sh start
#     bash scripts/gotrack_daemons.sh stop
#     bash scripts/gotrack_daemons.sh status
#     bash scripts/gotrack_daemons.sh log capture1   # tail one PC's log
set -euo pipefail

PCS=(capture1 capture2 capture3 capture5 capture6)  # capture4 out
# Resolved on the REMOTE side via $HOME — do NOT expand ~ locally.
PY='$HOME/anaconda3/envs/gotrack_cu128/bin/python'
DAEMON='$HOME/AutoDex/src/execution/daemon/gotrack_daemon.py'
LOG=/tmp/gotrack_daemon.log
# Capture PCs subscribe to the robot-host prior-pose PUB endpoint.  The old
# fixed default (192.168.0.2) belongs to the previous lab host and leaves the
# daemons waiting forever when this host has a different camera-LAN address.
# Let an explicit ROBOT_IP override this discovery for unusual topologies;
# otherwise use the source address selected by the route to capture1.
detect_robot_ip() {
    if [[ -n "${ROBOT_IP:-}" ]]; then
        printf '%s\n' "$ROBOT_IP"
        return 0
    fi

    local capture_ip route_ip
    capture_ip="$(getent ahostsv4 "${PCS[0]}" 2>/dev/null | awk 'NR == 1 {print $1}')"
    if [[ -z "$capture_ip" ]]; then
        echo "[gotrack] cannot resolve ${PCS[0]}; set ROBOT_IP explicitly" >&2
        return 1
    fi
    route_ip="$(ip route get "$capture_ip" 2>/dev/null | awk '{for (i = 1; i <= NF; ++i) if ($i == "src") {print $(i + 1); exit}}')"
    if [[ -z "$route_ip" ]]; then
        echo "[gotrack] cannot determine source IP for $capture_ip; set ROBOT_IP explicitly" >&2
        return 1
    fi
    printf '%s\n' "$route_ip"
}

ACTION="${1:-status}"

case "$ACTION" in
    start)
        ROBOT_IP="$(detect_robot_ip)"
        echo "[gotrack] robot prior-pose IP: $ROBOT_IP"
        for pc in "${PCS[@]}"; do
            ssh -o ConnectTimeout=3 "$pc" "pkill -9 -f gotrack_daemon 2>/dev/null || true" &
        done
        wait
        sleep 2
        # xformers Blackwell kernels need fp16/bf16 input. gotrack_engine wraps
        # the forward in torch.autocast(bf16) so xformers is active again.
        for pc in "${PCS[@]}"; do
            ssh -o ConnectTimeout=3 "$pc" "bash -c 'nohup $PY $DAEMON --robot-ip $ROBOT_IP > $LOG 2>&1 &'"
        done
        sleep 3
        for pc in "${PCS[@]}"; do
            n=$(ssh -o ConnectTimeout=3 "$pc" "pgrep -fc 'python.*gotrack_daemon'" 2>/dev/null || echo 0)
            echo "  $pc: $n daemon(s)"
        done
        ;;
    stop)
        for pc in "${PCS[@]}"; do
            ssh -o ConnectTimeout=3 "$pc" "pkill -9 -f gotrack_daemon 2>/dev/null && echo killed || true" &
        done
        wait
        ;;
    status)
        for pc in "${PCS[@]}"; do
            n=$(ssh -o ConnectTimeout=3 "$pc" "pgrep -fc 'python.*gotrack_daemon'" 2>/dev/null || echo "?")
            echo "  $pc: $n"
        done
        ;;
    log)
        pc="${2:-capture1}"
        ssh "$pc" "tail -50 $LOG"
        ;;
    *)
        echo "usage: $0 {start|stop|status|log [pc_name]}"
        exit 1
        ;;
esac
