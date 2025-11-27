# tests/test_moondream.py
from __future__ import annotations
from typing import Any, Optional
import os
import time
import shutil
from pathlib import Path
import glob

# === 1) Point to your fully local model folder ===
MODEL_DIR = Path("/home/team15/Documents/SmartGlass/moondream-2b-2025-04-14-4bit").resolve()

# === 2) Force offline + project-local cache so this works from ANY shell ===
CACHE_ROOT = (MODEL_DIR.parent / ".hf_cache").resolve()
os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# (Optional CUDA tweaks)
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
# os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

# --- deps ---
import torch  # type: ignore
from transformers import AutoModelForCausalLM  # type: ignore
from transformers.dynamic_module_utils import (  # type: ignore
    HF_MODULES_CACHE,
    TRANSFORMERS_DYNAMIC_MODULE_NAME,
)

# Pillow is optional; only needed if you call describe_image()
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore

# === 3) Monkey-patch tokenizers to use local tokenizer.json when Moondream asks the Hub ===
# Moondream’s code calls: tokenizers.Tokenizer.from_pretrained("vikhyatk/moondream2", revision="YYYY-MM-DD")
# In offline mode that fails unless we redirect to our local tokenizer.json.
from tokenizers import Tokenizer as _HFTokenizer  # type: ignore

_ORIG_FROM_PRETRAINED = _HFTokenizer.from_pretrained  # original classmethod

def _local_from_pretrained(cls, identifier, *args, **kwargs):
    """
    Correct classmethod signature: (cls, identifier, *args, **kwargs)
    Intercept only the Moondream hub id and return a Tokenizer from the local file.
    """
    # Normalize identifier to str (HF passes str here)
    if isinstance(identifier, str) and identifier.strip().lower() == "vikhyatk/moondream2":
        local_tok = MODEL_DIR / "tokenizer.json"
        if not local_tok.exists():
            raise FileNotFoundError(f"[moondream] Expected local tokenizer at {local_tok} but it was not found.")
        return cls.from_file(str(local_tok))  # use local tokenizer.json, entirely offline

    # Fallback to original behavior for any other identifier
    # Call the original *classmethod* correctly, passing cls explicitly:
    return _ORIG_FROM_PRETRAINED.__func__(cls, identifier, *args, **kwargs)

# Apply patch
_HFTokenizer.from_pretrained = classmethod(_local_from_pretrained)  # type: ignore

# === 4) Helper: pre-populate Transformers dynamic-module cache for remote code ===
def _prepopulate_dynamic_module_cache(model_dir: Path) -> Path:
    """
    Transformers remote-code loader compiles/imports from:
      ~/.cache/huggingface/modules/transformers_modules/<namespace>/
    The <namespace> is the basename of model_dir with '-' -> '_hyphen_'.
    Mirror all top-level .py files there so relative imports resolve.
    """
    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    namespace = model_dir.name.replace("-", "_hyphen_")
    dst = Path(HF_MODULES_CACHE) / TRANSFORMERS_DYNAMIC_MODULE_NAME / namespace
    dst.mkdir(parents=True, exist_ok=True)

    py_files = sorted(model_dir.glob("*.py"))
    if not py_files:
        raise FileNotFoundError(
            f"No Python files found in {model_dir}. "
            "Ensure hf_moondream.py and its siblings (layers.py, vision.py, etc.) exist."
        )

    for src in py_files:
        shutil.copy2(src, dst / src.name)
    (dst / "__init__.py").touch()
    return dst

# === 5) Loader + tiny API ===
_MODEL: Optional[Any] = None

def get_model() -> Any:
    """
    Load Moondream completely locally (no network):
      - uses MODEL_DIR
      - trust_remote_code=True
      - tokenizers patched to local tokenizer.json
      - dynamic-module cache pre-populated for relative imports
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    # quick sanity for required files
    required = ["config.json", "model.safetensors", "hf_moondream.py", "tokenizer.json"]
    missing = [n for n in required if not (MODEL_DIR / n).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {MODEL_DIR}: {missing}")

    cache_ns = _prepopulate_dynamic_module_cache(MODEL_DIR)
    print(f"[Moondream] populated: {cache_ns}")

    t0 = time.time()
    print(f"[Moondream] loading model from {MODEL_DIR} ...")
    max_memory = {
        0: "4.0GiB",   # GPU 0 limit
        "cpu": "16GiB" # optional, for CPU offload
    }
    _MODEL = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto" if torch.cuda.is_available() else None,  # GPU if present
        max_memory=max_memory
    )
    print(f"[Moondream] model ready in {time.time() - t0:.2f}s")
    return _MODEL

def get_image(img_path: str) -> Any:
    if Image is None:
        raise RuntimeError("Pillow (PIL) is not installed; cannot load images.")
    p = Path(img_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return Image.open(p)

def describe_image(image: Any, prompt: str) -> str:
    """
    Moondream exposes .query(image, prompt) via its remote-code module.
    """
    model = get_model()
    t0 = time.time()
    out = model.query(image, prompt)
    ans = out.get("answer", "").strip() if isinstance(out, dict) else str(out)
    print(f"[infer] {time.time() - t0:.2f}s")
    return ans

# === 6) CLI harness ===
def main():
    model = get_model()
    nparams = sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", False))
    print(f"[sanity] trainable params: {nparams}")

    if Image is None:
        print("[note] PIL not installed; skipping interactive image demo.")
        return

    print(f"\nMoondream ready @ {MODEL_DIR}\nType 'quit' to exit.\n")
    last: Optional[str] = None
    try:
        while True:
            img_path = input("Image path (or ENTER to reuse last): ").strip()
            if img_path.lower() in {"q", "quit", "exit"}:
                break
            if not img_path:
                if not last:
                    print("No previous image; please enter a path.\n")
                    continue
                img_path = last
            if not Path(img_path).exists():
                print(f"[!] Not found: {img_path}\n")
                continue
            prompt = input("Prompt: ").strip() or "Describe the image."
            if prompt.lower() in {"q", "quit", "exit"}:
                break

            img = get_image(img_path)
            reply = describe_image(img, prompt)
            print("\n--- RESPONSE ---")
            print(reply, "\n")
            last = img_path
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
