#!/bin/bash
# Per-object BODex grasp gen (box/shelf/wall) -> tar -> NAS backup -> erase local.
# Resumable: skips objects already backed up on NAS. Keeps local disk from filling
# (the cause of the earlier crash: / was ~95% full).
set -uo pipefail
cd "$(dirname "$0")"

PYTHON=/home/mingi/miniconda3/envs/bodex/bin/python
OBJ_LIST="$(dirname "$0")/../obj_list.txt"
OUT="$HOME/AutoDex/bodex_outputs/allegro/v8"
NAS="$HOME/shared_data/AutoDex/bodex_outputs/allegro/v8"
OBJ_ROOT="$HOME/shared_data/object_processing"
CONFIGS=(box shelf wall)

export TORCH_CUDA_ARCH_LIST=8.6   # avoid cuRobo JIT arch hang
mkdir -p "$NAS"
TMPLIST=$(mktemp); trap 'rm -f "$TMPLIST"' EXIT

while read -r obj; do
  [ -z "$obj" ] && continue
  if [ -f "$NAS/$obj.tar.gz" ]; then echo "[skip] $obj (already on NAS)"; continue; fi

  echo "[gen] $obj"
  echo "$obj" > "$TMPLIST"
  for cfg in "${CONFIGS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 $PYTHON generate.py -c "sim_allegro/paradex_${cfg}.yml" -w 20 \
      --exp_name v8 --obj_root_dir "$OBJ_ROOT" --obj_list_file "$TMPLIST" </dev/null \
      || { echo "[FAIL gen] $obj $cfg"; exit 1; }
  done

  if [ ! -d "$OUT/$obj" ]; then echo "[warn] no output for $obj"; continue; fi
  echo "[backup] $obj"
  tar czf "$OUT/$obj.tar.gz" -C "$OUT" "$obj" || { echo "[FAIL tar] $obj"; exit 1; }
  cp "$OUT/$obj.tar.gz" "$NAS/$obj.tar.gz"     || { echo "[FAIL cp] $obj"; exit 1; }
  if [ "$(stat -c%s "$OUT/$obj.tar.gz")" -eq "$(stat -c%s "$NAS/$obj.tar.gz")" ]; then
    rm -rf "$OUT/$obj" "$OUT/$obj.tar.gz"
    echo "[done] $obj (backed up + local erased)"
  else
    echo "[FAIL verify] $obj — keeping local, stopping"; exit 1
  fi
done < "$OBJ_LIST"
echo "ALL DONE"
