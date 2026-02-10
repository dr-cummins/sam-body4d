#!/usr/bin/env bash
# Sync source files from main repo to hf-space, excluding binaries.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_REPO="$(dirname "$SCRIPT_DIR")"
HF_REPO="${HF_REPO:-/Users/thomascummins/dev/chromatica/projects/hf-space}"

if [ ! -d "$HF_REPO" ]; then
    echo "ERROR: hf-space repo not found at $HF_REPO"
    exit 1
fi

echo "Syncing $MAIN_REPO → $HF_REPO"

# --- app.py ---
cp "$MAIN_REPO/app.py" "$HF_REPO/app.py"

# --- utils/ ---
rsync -a --delete --include='*.py' --exclude='*' \
    "$MAIN_REPO/utils/" "$HF_REPO/utils/"

# --- configs/ ---
# (body4d.yaml is written separately with ckpt_root override; skip here)

# --- scripts/setup.py ---
mkdir -p "$HF_REPO/scripts"
cp "$MAIN_REPO/scripts/setup.py" "$HF_REPO/scripts/setup.py"

# --- models/sam3/ (Python source + BPE vocab) ---
rsync -a --delete \
    --include='*/' \
    --include='*.py' \
    --include='pyproject.toml' \
    --include='setup.cfg' \
    --include='assets/bpe_simple_vocab_16e6.txt.gz' \
    --exclude='*.egg-info/' \
    --exclude='scripts/' \
    --exclude='.github/' \
    --exclude='tests/' \
    --exclude='*.png' --exclude='*.jpg' --exclude='*.gif' \
    --exclude='*.mp4' --exclude='*.avi' --exclude='*.mov' \
    --exclude='*.pt' --exclude='*.pth' --exclude='*.safetensors' \
    --exclude='*.ipynb' \
    --exclude='*' \
    "$MAIN_REPO/models/sam3/" "$HF_REPO/models/sam3/"

# --- models/sam_3d_body/ (Python source, exclude notebook binaries) ---
rsync -a --delete \
    --include='*/' \
    --include='*.py' \
    --include='*.yaml' \
    --include='*.json' \
    --exclude='notebook/images/' \
    --exclude='notebook/masks/' \
    --exclude='notebook/*.ipynb' \
    --exclude='*.png' --exclude='*.jpg' --exclude='*.gif' \
    --exclude='*.mp4' --exclude='*.avi' --exclude='*.mov' \
    --exclude='*.pt' --exclude='*.pth' --exclude='*.safetensors' \
    --exclude='*' \
    "$MAIN_REPO/models/sam_3d_body/" "$HF_REPO/models/sam_3d_body/"

# --- models/diffusion_vas/ (Python source, exclude binary dirs) ---
rsync -a --delete \
    --include='*/' \
    --include='*.py' \
    --include='*.yaml' \
    --include='*.json' \
    --exclude='models/Depth_Anything_V2/assets/' \
    --exclude='models/Depth_Anything_V2/metric_depth/dataset/splits/' \
    --exclude='*.png' --exclude='*.jpg' --exclude='*.gif' \
    --exclude='*.mp4' --exclude='*.avi' --exclude='*.mov' \
    --exclude='*.pt' --exclude='*.pth' --exclude='*.safetensors' \
    --exclude='*.ipynb' \
    --exclude='*' \
    "$MAIN_REPO/models/diffusion_vas/" "$HF_REPO/models/diffusion_vas/"

# --- Remove __init__.py that would break namespace package resolution ---
# models/ and models/diffusion_vas/ MUST be namespace packages (no __init__.py)
# so that Python can merge paths across sys.path entries. diffusion_vas/demo.py
# imports from models.diffusion_vas.pipeline_diffusion_vas which lives in a
# NESTED models/diffusion_vas/models/diffusion_vas/ directory.
rm -f "$HF_REPO/models/__init__.py"
rm -f "$HF_REPO/models/diffusion_vas/__init__.py"

echo "Sync complete."
echo ""
echo "Files synced:"
find "$HF_REPO" -name '*.py' -o -name '*.yaml' -o -name '*.json' -o -name '*.toml' | \
    grep -v '.git/' | wc -l | xargs echo "  Total text files:"
