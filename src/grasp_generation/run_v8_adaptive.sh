#!/bin/bash
# v8 grasp generation via the adaptive orchestrator (the v7 method):
# per (obj, scene_type, scene_id) escalate gap [0.02..0.08] x seed_num [200,1000]
# until >=5 sim-filter-passing grasps; on success at 0.02 try bonus gap 0.0.
#
# The ONLY thing added on top of the orchestrator is per-object NAS backup:
#   candidates -> NAS (rsync), bodex_outputs -> NAS (tar.gz) -> erase local bodex
# to keep the local disk from filling. The grasp-generation algorithm itself
# (adaptive_orchestrator.py) is called EXACTLY as before, unchanged.
#
# Resume: a per-object marker under {NAS_BODEX}/.done/{obj} means "fully done +
# backed up". Restarting skips marked objects. --resume also lets the
# orchestrator continue a partially-done object.
#
# IMPORTANT: the orchestrator must be called PER OBJECT (--obj). The BODex
# scene_filter is scene-id-only and applies to EVERY object in obj_list, so
# passing the full obj_list makes each object's turn regenerate ALL objects.
#
# Usage: run_v8_adaptive.sh [allegro|inspire|inspire_left|all]
set -uo pipefail
cd "$(dirname "$0")" || exit 1

# cuRobo JIT: pin to this GPU's arch (RTX 3090 = 8.6) so the BODex subprocess
# reuses the cached .so instead of an all-arch rebuild that hangs.
export TORCH_CUDA_ARCH_LIST="8.6"

PYTHON=/home/mingi/miniconda3/envs/mingi/bin/python
OBJ_LIST="$(pwd)/obj_list_v8.txt"
OBJ_ROOT="$HOME/shared_data/object_processing"
LOGDIR="$HOME/AutoDex/logging/adaptive"
LOCAL_BODEX_BASE="$HOME/AutoDex/bodex_outputs"
LOCAL_CAND_BASE="$HOME/AutoDex/candidates"
NAS_BODEX_BASE="$HOME/shared_data/AutoDex/bodex_outputs"
NAS_CAND_BASE="$HOME/shared_data/AutoDex/candidates"
mkdir -p "$LOGDIR"

HANDS_ARG="${1:-all}"
case "$HANDS_ARG" in
  all) HANDS=(allegro inspire inspire_left) ;;
  allegro|inspire|inspire_left) HANDS=("$HANDS_ARG") ;;
  *) echo "usage: $0 [allegro|inspire|inspire_left|all]"; exit 1 ;;
esac

mapfile -t OBJS < <(grep -v '^#' "$OBJ_LIST" | sed '/^[[:space:]]*$/d')
echo "objects: ${#OBJS[@]}  hands: ${HANDS[*]}"

for hand in "${HANDS[@]}"; do
  echo "[$(date '+%F %T')] ===== v8 adaptive hand=$hand ====="
  NAS_BODEX="$NAS_BODEX_BASE/$hand/v8"
  NAS_CAND="$NAS_CAND_BASE/$hand/v8"
  DONE_DIR="$NAS_BODEX/.done"
  mkdir -p "$NAS_BODEX" "$NAS_CAND" "$DONE_DIR"

  for obj in "${OBJS[@]}"; do
    [ -z "$obj" ] && continue
    if [ -f "$DONE_DIR/$obj" ]; then echo "[skip] $obj (done)"; continue; fi
    echo "[$(date '+%F %T')] --- $hand / $obj ---"

    # ===== orchestration: UNCHANGED =====
    "$PYTHON" adaptive_orchestrator.py \
      --hand "$hand" --version v8 \
      --obj "$obj" \
      --scenes wall shelf box \
      --obj_root "$OBJ_ROOT" \
      --resume 2>&1 | tee -a "$LOGDIR/run_v8_${hand}.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "[FAIL orch] $hand/$obj rc=$rc — reclaim partial bodex, continue"
      rm -rf "$LOCAL_BODEX_BASE/$hand/v8/$obj"
      continue
    fi

    # ===== added: back up to NAS as tar.gz (single-file writes are fast on the
    # NFS; thousands of small-file writes are not). Candidates STAY local
    # (small, browsable, needed by downstream order/viewers); only the huge raw
    # bodex_outputs is erased to protect disk. =====
    CAND_V8="$LOCAL_CAND_BASE/$hand/v8"
    BODEX_V8="$LOCAL_BODEX_BASE/$hand/v8"

    # candidates -> NAS tarball (keep local dir)
    if [ -d "$CAND_V8/$obj" ]; then
      if tar czf "$CAND_V8/$obj.tar.gz" -C "$CAND_V8" "$obj" \
         && cp "$CAND_V8/$obj.tar.gz" "$NAS_CAND/$obj.tar.gz" \
         && [ "$(stat -c%s "$CAND_V8/$obj.tar.gz")" -eq "$(stat -c%s "$NAS_CAND/$obj.tar.gz")" ]; then
        rm -f "$CAND_V8/$obj.tar.gz"
      else
        echo "[FAIL cand-backup] $hand/$obj — not marking done"
        rm -f "$CAND_V8/$obj.tar.gz"
        continue
      fi
    fi

    # bodex_outputs -> NAS tarball + erase local (disk protection)
    if [ -d "$BODEX_V8/$obj" ]; then
      if tar czf "$BODEX_V8/$obj.tar.gz" -C "$BODEX_V8" "$obj" \
         && cp "$BODEX_V8/$obj.tar.gz" "$NAS_BODEX/$obj.tar.gz" \
         && [ "$(stat -c%s "$BODEX_V8/$obj.tar.gz")" -eq "$(stat -c%s "$NAS_BODEX/$obj.tar.gz")" ]; then
        rm -rf "$BODEX_V8/$obj" "$BODEX_V8/$obj.tar.gz"
      else
        echo "[FAIL bodex-backup] $hand/$obj — keeping local, not marking done"
        rm -f "$BODEX_V8/$obj.tar.gz"
        continue
      fi
    fi

    touch "$DONE_DIR/$obj"
    echo "[done] $hand/$obj (candidates local+NAS tar, bodex NAS tar + local erased)"
  done
  echo "[$(date '+%F %T')] ===== hand=$hand done ====="
done
echo "[$(date '+%F %T')] v8 adaptive run complete: ${HANDS[*]}"
