#!/usr/bin/env python3
# Interactive scene description using local Moondream VLM.
import os
import sys
import socket
import struct
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from jetson_utils import loadImage, cudaFromNumpy
    _HAS_JETSON_UTILS = True
except Exception:
    _HAS_JETSON_UTILS = False

HOST = "192.168.4.1"
PORT = 2000

TMP_DIR = Path("temp")
TMP_DIR.mkdir(parents=True, exist_ok=True)
FRAME_PATH = TMP_DIR / "esp_frame.jpg"
_MODEL: Optional[AutoModelForCausalLM] = None

def recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    # Read exactly n bytes from socket
    buf = b""
    while len(buf) < n:
        pkt = sock.recv(n - len(buf))
        if not pkt:
            return None
        buf += pkt
    return buf

def recv_one_frame(sock: socket.socket) -> Optional[Image.Image]:
    # Read MJPEG: 4-byte big-endian length + JPEG payload
    hdr = recvall(sock, 4)
    if hdr is None:
        return None
    (length,) = struct.unpack(">I", hdr)
    data = recvall(sock, length)
    if data is None:
        return None
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return Image.fromarray(img)

def get_model() -> AutoModelForCausalLM:
    # Load Moondream VLM model (cached)
    global _MODEL
    if _MODEL is None:
        import time
        t0 = time.time()
        _MODEL = AutoModelForCausalLM.from_pretrained(
            "moondream/moondream-2b-2025-04-14-4bit",
            trust_remote_code=True,
            device_map={"": "cuda"}
        )
        dt = time.time() - t0
        print(f"Model loaded in {dt:.2f}s")
    return _MODEL

def describe_image(image: Image.Image, prompt: str) -> str:
    # Query VLM for image description
    import time
    model = get_model()
    t0 = time.time()
    out = model.query(image, prompt)
    ans = out.get("answer", "").strip() if isinstance(out, dict) else str(out)
    dt = time.time() - t0
    print(f"Inference: {dt:.2f}s")
    return ans
    
def main():
    # Interactive scene description loop
    import time
    _ = get_model()
    print("Ready. Press 'r' to capture, 'q' to quit.\n")

    try:
        with socket.create_connection((HOST, PORT)) as s:
            print(f"Connected to ESP at {HOST}:{PORT}")
            while True:
                try:
                    cmd = input(">> ").strip().lower()
                except EOFError:
                    break

                if cmd in ("q", "quit", "exit"):
                    break
                if cmd != "r":
                    print("Type 'r' to capture and describe one frame, or 'q' to quit.")
                    continue

                # Receive exactly one frame
                frame = recv_one_frame(s)
                if frame is None:
                    print("[!] Could not receive a frame. Exiting.")
                    break

                # Save to disk (overwrite same path)
                ok = cv2.imwrite(str(FRAME_PATH), frame)
                if not ok:
                    print(f"[!] Failed to write frame to {FRAME_PATH}")
                    continue

                # Describe
                try:
                    reply = describe_image(frame, "Describe the image.")
                    print("\n--- RESPONSE ---")
                    print(reply, "\n")
                except Exception as e:
                    print(f"[!] Inference error: {e}\n")

    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        print(f"[!] Could not connect to {HOST}:{PORT}: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
