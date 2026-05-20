#!/bin/bash
# Download DINOv3 weights from Meta's CDN.
#
# IMPORTANT: Meta gates these URLs behind their DINOv3 License Agreement.
# Without a valid Meta token, the URLs return HTTP 403.
# Steps to get access:
#   1. Go to https://github.com/facebookresearch/dinov3
#   2. Click the "Request access" / license-accept link in the README
#   3. Either configure your Meta credentials, OR follow the manual download links Meta sends you
#   4. Drop the downloaded .pth files into src/models/weights/
#
# Once weights are local, register them in src/models/dinov3_loader.py:_AVAILABLE_WEIGHTS.
#
# The hashes below are extracted verbatim from
# src/models/dinov3_repo/dinov3/hub/backbones.py (Meta's official codebase).

set -e

cd "$(dirname "$0")/.."
WEIGHTS_DIR="src/models/weights"
mkdir -p "$WEIGHTS_DIR"

BASE_URL="https://dl.fbaipublicfiles.com/dinov3"

declare -A WEIGHTS=(
    ["dinov3_vits16"]="08c60483"
    ["dinov3_vits16plus"]="4057cbaa"
    ["dinov3_vitb16"]="73cec8be"
    ["dinov3_vitl16"]="8aa4cbdd"
    ["dinov3_vitl16plus"]="46503df0"
    ["dinov3_vith16plus"]="7c1da9a5"
)

ARCHES_TO_FETCH=("${@:-dinov3_vitl16plus dinov3_vith16plus}")

for arch in $ARCHES_TO_FETCH; do
    hash="${WEIGHTS[$arch]:-}"
    if [ -z "$hash" ]; then
        echo "ERROR: unknown arch '$arch'. Available: ${!WEIGHTS[*]}"
        continue
    fi
    filename="${arch}_pretrain_lvd1689m-${hash}.pth"
    dest="${WEIGHTS_DIR}/${filename}"
    url="${BASE_URL}/${arch}/${filename}"

    if [ -f "$dest" ]; then
        echo "✓ Already present: $dest"
        continue
    fi

    echo "↓ Downloading $arch ..."
    echo "   URL: $url"
    if wget -q --show-progress -O "$dest" "$url"; then
        size=$(du -h "$dest" | cut -f1)
        echo "✓ Downloaded $dest ($size)"
    else
        rm -f "$dest"
        echo "✗ FAILED ($arch). Meta gates this URL — see the script header for instructions."
        echo "   You can manually download the file and place it at: $dest"
    fi
done

echo ""
echo "Currently registered weights in dinov3_loader.py:"
grep -E "^\s*\"dinov3_" src/models/dinov3_loader.py | head -20 || true
echo ""
echo "After downloading manually, register each path in _AVAILABLE_WEIGHTS in src/models/dinov3_loader.py."
