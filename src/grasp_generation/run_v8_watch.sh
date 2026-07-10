#!/bin/bash
# Keep run_v8_adaptive.sh alive overnight: if it dies before "run complete",
# restart it. The per-object .done markers make restarts resume cleanly.
# pgrep guard prevents double-launch.
#
# Usage: run_v8_watch.sh [allegro|inspire|inspire_left|all]   (default allegro)
cd "$(dirname "$0")" || exit 1
HAND="${1:-allegro}"
LOG="$HOME/AutoDex/logging/adaptive/run_v8_watch_${HAND}.log"
mkdir -p "$(dirname "$LOG")"

while ! grep -q "v8 adaptive run complete" "$LOG" 2>/dev/null; do
  if ! pgrep -f "[r]un_v8_adaptive.sh" >/dev/null; then
    echo "[watcher] restart $(date)" >> "$LOG"
    bash run_v8_adaptive.sh "$HAND" >> "$LOG" 2>&1
  fi
  sleep 60
done
echo "[watcher] complete, exiting $(date)" >> "$LOG"
