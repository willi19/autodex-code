#!/bin/bash
# Download model weights from NAS to local thirdparty/weights/
# Run once after git clone.
#
# Usage: bash scripts/setup_weights.sh

set -e

NAS_WEIGHTS="/home/$(whoami)/shared_data/AutoDex/weights"
LOCAL_WEIGHTS="$(dirname "$0")/../autodex/perception/thirdparty/weights"

mkdir -p "$LOCAL_WEIGHTS"

echo "Copying weights from NAS..."

# FoundationPose
echo "  FoundationPose..."
mkdir -p "$LOCAL_WEIGHTS/foundationpose"
cp -r "$NAS_WEIGHTS/foundationpose/"* "$LOCAL_WEIGHTS/foundationpose/"

# SAM3 (downloaded via HuggingFace on first run, but copy if available)
if [ -d "$NAS_WEIGHTS/sam3" ]; then
    echo "  SAM3..."
    mkdir -p "$LOCAL_WEIGHTS/sam3"
    cp -r "$NAS_WEIGHTS/sam3/"* "$LOCAL_WEIGHTS/sam3/"
fi

# DA3 (downloaded via HuggingFace on first run, but copy if available)
if [ -d "$NAS_WEIGHTS/da3" ]; then
    echo "  DA3..."
    mkdir -p "$LOCAL_WEIGHTS/da3"
    cp -r "$NAS_WEIGHTS/da3/"* "$LOCAL_WEIGHTS/da3/"
fi

# YOLOE
if [ -f "$NAS_WEIGHTS/yoloe/yoloe-26x-seg.pt" ]; then
    echo "  YOLOE..."
    # YoloeSegmentor resolves this exact flat path via YOLOE_WEIGHTS.
    cp "$NAS_WEIGHTS/yoloe/yoloe-26x-seg.pt" "$LOCAL_WEIGHTS/yoloe-26x-seg.pt"
fi

echo "Done. Weights at: $LOCAL_WEIGHTS"
ls -lh "$LOCAL_WEIGHTS/"
