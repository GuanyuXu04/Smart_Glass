# tests/test_load_moondream.py
import os
import sys
import time
import shutil
import glob
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM
    from transformers.dynamic_module_utils import HF_MODULES_CACHE, TRANSFORMERS_DYNAMIC_MODULE_NAME
except Exception as e:
    raise RuntimeError(
        "Transformers not installed in this environment. "
        "Run via scripts/run_moondream_local.sh"
    ) from e

MODEL_PATH = "/home/team15/Documents/SmartGlass/moondream-2b-2025-04-14-4bit"

def _prepopulate_dynamic_module_cache(model_dir: str) -> str:
    """
    Mirror the model's top-level .py files into the Transformers
    dynamic-module cache under the namespace derived from the folder name.
    This avoids '.../transformers_modules/<ns>/layers.py not found' errors.
    """
    model_dir = os.path.abspath(model_dir)
    base = os.path.basename(model_dir)
    ns = base.replace("-", "_hyphen_")  # mirrors transformers' mapping
    dst = os.path.join(HF_MODULES_CACHE, TRANSFORMERS_DYNAMIC_MODULE_NAME, ns)
    os.makedirs(dst, exist_ok=True)

    py_files = sorted(glob.glob(os.path.join(model_dir, "*.py")))
    for src in py_files:
        shutil.copy2(src, dst)
    Path(os.path.join(dst, "__init__.py")).touch()

    return dst

def test_load_local_model():
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"Model folder not found: {MODEL_PATH}")

    # Strong offline stance and project-local cache defaults; harmless if already set.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(Path(MODEL_PATH).parent / ".hf_cache"))

    # Ensure required entry file is present (per config auto_map)
    entry_file = Path(MODEL_PATH, "hf_moondream.py")
    if not entry_file.exists():
        raise FileNotFoundError(f"Missing entry module: {entry_file}")

    # Pre-populate the dynamic-modules cache for this folder name.
    cache_ns_path = _prepopulate_dynamic_module_cache(MODEL_PATH)
    print(f"[INFO] Prepopulated dynamic-module cache at: {cache_ns_path}")

    print(f"[INFO] Attempting to load model from: {MODEL_PATH}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,    # loads entry module and resolves relative imports
        local_files_only=True,     # never touch the network
        # Let Transformers default to CPU if CUDA not available.
        # You can force CPU by setting device_map=None.
        device_map="auto",
    )
    dt = time.time() - t0
    print(f"[INFO] Model loaded successfully in {dt:.2f} seconds")

    # Very light sanity check
    num_params = sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", False))
    print(f"[INFO] Number of trainable params: {num_params}")
    assert num_params > 0, "Loaded model has zero trainable parameters (unexpected)"

    print("[INFO] Local model load test passed ✅")

if __name__ == "__main__":
    test_load_local_model()
