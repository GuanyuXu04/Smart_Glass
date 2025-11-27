#!/usr/bin/env bash
set -euo pipefail

# === paths you may edit ===
PROJECT_DIR="$HOME/Documents/SmartGlass"
MODEL_DIR="$PROJECT_DIR/moondream-2b-2025-04-14-4bit"   # your local model folder (already copied & de-symlinked)
PYTHON_VERSION="${PYTHON_VERSION:-python3}"              # or python3.10, etc.
VENV_DIR="$PROJECT_DIR/.venv_moondream"
HF_HOME_DIR="$PROJECT_DIR/.hf_cache"                     # project-local HF cache dir
# ==========================

echo "[setup] Creating venv at: $VENV_DIR"
$PYTHON_VERSION -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel

# Lightweight deps. If torch is already system-wide, we won't re-install.
if ! python -c 'import torch' >/dev/null 2>&1; then
  echo "[setup] Installing torch (CPU build)."
  pip install --upgrade "torch>=2.2,<3"                   # CPU build via PyPI
fi

# Transformers with remote-code support
pip install --upgrade "transformers>=4.46,<5" "huggingface_hub>=0.22" pillow "torchao>=0.10.0" accelerate

# Strictly offline + project-local caches
export HF_HOME="$HF_HOME_DIR"
export TRANSFORMERS_CACHE="$HF_HOME/models"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Pre-populate the Transformers dynamic-module cache for this *exact* folder name.
python - <<'PY'
import os, shutil, glob, pathlib
from transformers.dynamic_module_utils import HF_MODULES_CACHE, TRANSFORMERS_DYNAMIC_MODULE_NAME
# These must match the bash variables above:
MODEL_DIR = os.environ.get("MODEL_DIR_PY") or "/home/team15/Documents/SmartGlass/moondream-2b-2025-04-14-4bit"

# Transformers maps basename -> dynamic namespace; hyphens become "_hyphen_"
base = os.path.basename(MODEL_DIR)
ns = base.replace("-", "_hyphen_")

cache_root = os.path.join(HF_MODULES_CACHE, TRANSFORMERS_DYNAMIC_MODULE_NAME)
dst = os.path.join(cache_root, ns)
os.makedirs(dst, exist_ok=True)

# copy all top-level .py model files into the cache (so relative imports resolve there)
py_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.py")))
if not py_files:
    raise SystemExit(f"No .py files found in model dir: {MODEL_DIR}")
for src in py_files:
    shutil.copy2(src, dst)

# make it a package
open(os.path.join(dst, "__init__.py"), "a").close()

print(f"[dynmod] Populated dynamic module cache: {dst}")
print("[dynmod] Files:", [os.path.basename(x) for x in py_files])
PY
# export path for the python block
export MODEL_DIR_PY="$MODEL_DIR"
