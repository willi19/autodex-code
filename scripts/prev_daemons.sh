#!/usr/bin/env bash
# Manage the legacy SAM3 + FoundationPose daemons used by the DA3 + FPose
# perception path (src/execution_prev/daemon/perception_daemon.py).
#
# These are NOT the current pipeline's daemons. init_daemon and snapshot_daemon
# already fill the capture PCs' 16 GB GPUs (~15.6 GB in use), so stop those
# first or SAM3/FPose will OOM on load:
#
#     bash scripts/daemons.sh stop
#     bash scripts/prev_daemons.sh check      # weights present everywhere?
#     bash scripts/prev_daemons.sh start
#
# Usage:
#     bash scripts/prev_daemons.sh start [sam3|fpose]
#     bash scripts/prev_daemons.sh stop  [sam3|fpose]
#     bash scripts/prev_daemons.sh status
#     bash scripts/prev_daemons.sh check                 # weights + envs
#     bash scripts/prev_daemons.sh log capture1
set -uo pipefail

# capture4 is gone: it answers no ssh and paradex's system config raises
# KeyError('capture4'). Do not put it back without checking both.
PCS_SAM3=(capture1 capture2 capture3)
PCS_FPOSE=(capture5 capture6)
PORT_SAM3=5001
PORT_FPOSE=5003

# Resolved on the REMOTE side via $HOME — do NOT expand ~ locally.
PY_SAM3='$HOME/anaconda3/envs/sam3/bin/python'
PY_FPOSE='$HOME/anaconda3/envs/foundationpose/bin/python'
DAEMON='$HOME/AutoDex/src/execution_prev/daemon/perception_daemon.py'
LOG_SAM3=/tmp/sam3_daemon.log
LOG_FPOSE=/tmp/fpose_daemon.log

ACTION="${1:-status}"
WHICH="${2:-}"

want() { [[ -z "$WHICH" || "$WHICH" == "$1" ]]; }

start_kind() {           # $1=kind $2=python $3=port $4=log  $5.. = pcs
    local kind=$1 py=$2 port=$3 log=$4; shift 4
    for pc in "$@"; do
        ssh -o ConnectTimeout=3 "$pc" \
            "pkill -9 -f 'perception_daemon.py --model $kind' 2>/dev/null || true" &
    done
    wait
    sleep 2
    for pc in "$@"; do
        # -n, setsid and closing all three fds: without them ssh stays open for
        # as long as the daemon lives and the loop never reaches the next PC.
        # nohup alone is not enough -- ssh waits on the fds, not the process.
        ssh -n -o ConnectTimeout=3 "$pc" \
            "cd \$HOME/AutoDex && setsid $py $DAEMON --model $kind --port $port \
                > $log 2>&1 < /dev/null &" > /dev/null 2>&1
    done
    # Model load is slow (SAM3 ~3.3 GB, FoundationPose builds its refiner), so
    # a daemon that is up but not yet bound is normal here. `status` is the
    # check that matters.
    sleep 5
    for pc in "$@"; do
        n=$(ssh -o ConnectTimeout=3 "$pc" \
            "ps -eo args | grep -c '[p]ython .*perception_daemon.py --model $kind' || true" 2>/dev/null || echo "?")
        echo "  $pc ($kind): $n"
    done
}

case "$ACTION" in
    start)
        want sam3  && start_kind sam3  "$PY_SAM3"  "$PORT_SAM3"  "$LOG_SAM3"  "${PCS_SAM3[@]}"
        want fpose && start_kind fpose "$PY_FPOSE" "$PORT_FPOSE" "$LOG_FPOSE" "${PCS_FPOSE[@]}"
        ;;
    stop)
        for pc in "${PCS_SAM3[@]}" "${PCS_FPOSE[@]}"; do
            ssh -o ConnectTimeout=3 "$pc" \
                "pkill -9 -f perception_daemon.py 2>/dev/null && echo killed || true" &
        done
        wait
        ;;
    status)
        for kind in sam3 fpose; do
            want "$kind" || continue
            if [[ "$kind" == sam3 ]]; then pcs=("${PCS_SAM3[@]}"); port=$PORT_SAM3
            else pcs=("${PCS_FPOSE[@]}"); port=$PORT_FPOSE; fi
            for pc in "${pcs[@]}"; do
                # The bracketed pattern keeps the remote shell from matching
                # itself; the port check is what the ZMQ client actually needs.
                out=$(ssh -o ConnectTimeout=3 "$pc" "
                    n=\$(ps -eo args | grep -c '[p]ython .*perception_daemon.py --model $kind' || true);
                    b=\$( (ss -ltn 2>/dev/null || netstat -ltn) | grep -c ':$port ' || true);
                    echo \"proc=\$n bound=\$b\"" 2>/dev/null || echo "unreachable")
                echo "  $pc ($kind): $out"
            done
        done
        ;;
    check)
        echo "Weights and envs (a missing one means: run scripts/setup_weights.sh there)"
        for kind in sam3 fpose; do
            # The conda env for fpose is named foundationpose, not fpose.
            if [[ "$kind" == sam3 ]]; then pcs=("${PCS_SAM3[@]}"); env_name=sam3
            else pcs=("${PCS_FPOSE[@]}"); env_name=foundationpose; fi
            for pc in "${pcs[@]}"; do
                out=$(ssh -o ConnectTimeout=3 "$pc" "
                    W=\$HOME/AutoDex/autodex/perception/thirdparty;
                    s=\$(ls \$W/weights/sam3/sam3.pt 2>/dev/null | wc -l);
                    f=\$(ls -d \$W/FoundationPose/weights/2023-10-28-18-33-37 2>/dev/null | wc -l);
                    m=\$(ls \$W/FoundationPose/mycpp/build/*.so 2>/dev/null | wc -l);
                    e=\$([ -x \$HOME/anaconda3/envs/$env_name/bin/python ] && echo y || echo n);
                    g=\$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1);
                    echo \"env=\$e sam3.pt=\$s fpose_ckpt=\$f mycpp=\$m gpu=[\$g]\"" 2>/dev/null \
                    || echo "unreachable")
                echo "  $pc ($kind): $out"
            done
        done
        ;;
    log)
        pc="${2:-capture1}"
        for log in "$LOG_SAM3" "$LOG_FPOSE"; do
            echo "===== $log @ $pc ====="
            ssh -o ConnectTimeout=3 "$pc" "tail -40 $log 2>/dev/null || echo '(no log)'"
        done
        ;;
    *)
        echo "usage: $0 {start|stop|status|check|log} [sam3|fpose | pc_name]"
        exit 1
        ;;
esac
