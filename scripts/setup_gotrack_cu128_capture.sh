#!/usr/bin/env bash
# Setup gotrack_cu128 env + MV-GoTrack on a capture PC.
# Run AS-IS on each capture PC after cloning AutoDex.
#
# Idempotent: re-running skips finished steps.
# Does NOT install gotrack_checkpoint.pt or anchor_banks/ — those are
# rsynced separately from robot PC (see scripts/sync_gotrack_assets.sh).
set -euo pipefail

REPO_ROOT="$HOME/AutoDex"
THIRDPARTY="$REPO_ROOT/autodex/perception/thirdparty"
GOTRACK_DIR="$THIRDPARTY/MV-GoTrack"
ENV_NAME="gotrack_cu128"
CONDA_DIR="$HOME/anaconda3"
PY="$CONDA_DIR/envs/$ENV_NAME/bin/python"
PIP="$CONDA_DIR/envs/$ENV_NAME/bin/pip"

echo "[gotrack-setup] host=$(hostname) repo=$REPO_ROOT"

# 1. Sync AutoDex
cd "$REPO_ROOT"
git fetch origin
git reset --hard origin/main

# 2. Clone MV-GoTrack (if missing)
mkdir -p "$THIRDPARTY"
cd "$THIRDPARTY"
if [ ! -d "MV-GoTrack/.git" ]; then
    git clone https://github.com/gunhee1113/MV-GoTrack.git
fi
cd "$GOTRACK_DIR"
git submodule update --init --recursive

# 3. Conda env (skip if exists)
if ! "$CONDA_DIR/bin/conda" env list | grep -qE "^$ENV_NAME\s"; then
    "$CONDA_DIR/bin/conda" create -n "$ENV_NAME" python=3.10 -y
fi

# 4. torch cu128 + transitive deps
$PIP install --force-reinstall torch torchvision xformers \
    --index-url https://download.pytorch.org/whl/cu128 --no-deps
$PIP install typing-extensions sympy networkx jinja2 filelock fsspec numpy
$PIP install "nvidia-nvshmem-cu12==3.4.5" "triton==3.6.0" pillow "setuptools<82"
$PIP install \
    "cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1" \
    "nvidia-cudnn-cu12==9.19.0.56" \
    "nvidia-cusparselt-cu12==0.7.1" \
    "nvidia-nccl-cu12==2.28.9"
$PIP install "cuda-bindings<13,>=12.9.4"

# 5. bop_toolkit (editable) + dinov2
cd "$GOTRACK_DIR"
$PIP install -e external/bop_toolkit
cd "$GOTRACK_DIR/external/dinov2"
$PY setup.py install
cd "$GOTRACK_DIR"

# 6. Pip deps + numpy pin (matplotlib pulls numpy 2.x; bop_toolkit needs <2)
$PIP install matplotlib distinctipy faiss-gpu-cu12 kornia pyrender pyglet \
    pyopengl imageio scikit-learn
$PIP install "numpy<2.0"

# 7. nvdiffrast (CRITICAL — silent pyrender fallback otherwise)
$PIP install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

# 8. FoundationPose extras + einops
$PIP install psutil pandas open3d transformations ruamel.yaml einops

# 9. GoTrack compatibility patches (idempotent — skip if already applied).
# The first keeps nvdiffrast FP32; the second restricts BF16 to the neural
# forward so Blackwell xFormers works without leaking BF16 into NumPy/PnP.
for PATCH in \
    "$REPO_ROOT/patches/MV-GoTrack-renderer-fix.patch" \
    "$REPO_ROOT/patches/MV-GoTrack-online-bf16-boundary.patch"; do
    if git apply --reverse --check "$PATCH" 2>/dev/null; then
        echo "[gotrack-setup] already applied: $(basename "$PATCH")"
    else
        git apply "$PATCH"
    fi
done

# 10. Sanity check
$PY -c "
import sys
sys.path.insert(0, '$GOTRACK_DIR')
from utils import renderer_nvdiffrast
import nvdiffrast.torch as dr
import torch
ctx = dr.RasterizeCudaContext()
print('OK', torch.cuda.get_device_name(0))
"

echo "[gotrack-setup] done on $(hostname)"
