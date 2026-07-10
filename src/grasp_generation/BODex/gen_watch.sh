#!/bin/bash
# Keep gen_backup.sh alive overnight: if it dies before "ALL DONE", restart it.
# gen_backup skips objects already on NAS, so a restart resumes cleanly.
# pgrep guard prevents double-launch while an instance is already running.
cd "$(dirname "$0")"
LOG="$HOME/AutoDex/logging/grasp_generation/gen_backup_v8.log"

while ! grep -q "ALL DONE" "$LOG" 2>/dev/null; do
  if ! pgrep -f "[g]en_backup.sh" >/dev/null; then
    echo "[watcher] restart $(date)" >> "$LOG"
    bash gen_backup.sh >> "$LOG" 2>&1
  fi
  sleep 60
done
echo "[watcher] ALL DONE, exiting $(date)" >> "$LOG"
