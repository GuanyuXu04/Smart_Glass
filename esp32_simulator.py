#!/usr/bin/env python3
"""ESP32 simulator for Smart Glass Jetson pipeline.

Provides 4 TCP services compatible with README contract:
  - 1000: Wake audio (WAV header + PCM stream)
  - 2000: MJPEG frames (length-prefixed JPEG)
  - 3000: MP3 sink (accepts bytes and discards)
  - 4000: Haptic commands sink (ASCII LLLRRR / XXXYYY)

Use this to exercise the pipeline on a dev machine without real hardware:
  python3 esp32_simulator.py --bind 127.0.0.1
Then run the pipeline with --host 127.0.0.1
"""
from __future__ import annotations

import argparse
import os
import socket
import struct
import threading
import time
from pathlib import Path
import io
import wave
import audioop

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # optional; video falls back to blank frames
    cv2 = None  # type: ignore
    np = None  # type: ignore

AUDIO_PORT = 1000
VIDEO_PORT = 2000
MP3_PORT = 3000
HAPTIC_PORT = 4000

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "images"
DEFAULT_IMG = IMG_DIR / "room.jpg"
SIM_WAV = ROOT / "audio" / "help_me_navigate_to_duderstadt_center.wav"

# ------------------------ Helpers ------------------------

def _make_wav_bytes(seconds: float = 2.0, sr: int = 16000, hz: float = 440.0):
    """Generate a mono 16-bit PCM WAV with a sine tone."""
    import math

    n_samples = int(seconds * sr)
    pcm = bytearray()
    for i in range(n_samples):
        s = int(0.3 * 32767 * math.sin(2 * math.pi * hz * (i / sr)))
        pcm.extend(struct.pack("<h", s))
    # WAV header (PCM, mono, 16-bit)
    subchunk1_size = 16
    audio_format = 1
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sr * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data = bytes(pcm)
    data_size = len(data)
    riff_chunk_size = 36 + data_size
    hdr = bytearray()
    hdr.extend(struct.pack("<4sI4s", b"RIFF", riff_chunk_size, b"WAVE"))
    hdr.extend(struct.pack("<4sI", b"fmt ", subchunk1_size))
    hdr.extend(struct.pack("<HHIIHH", audio_format, num_channels, sr, byte_rate, block_align, bits_per_sample))
    hdr.extend(struct.pack("<4sI", b"data", data_size))
    return bytes(hdr) + data

def _make_pcm_bytes(seconds: float = 2.0, sr: int = 16000, hz: float = 440.0) -> bytes:
    """Generate mono 16-bit PCM (no WAV header)."""
    import math
    n_samples = int(seconds * sr)
    pcm = bytearray()
    for i in range(n_samples):
        s = int(0.3 * 32767 * math.sin(2 * math.pi * hz * (i / sr)))
        pcm.extend(struct.pack("<h", s))  # 16-bit little-endian
    return bytes(pcm)


def _load_image_bytes() -> bytes:
    if cv2 is None:
        return b""  # will produce blank frames later
    path = DEFAULT_IMG if DEFAULT_IMG.exists() else next(iter(IMG_DIR.glob("*.jpg")), None)
    if path and path.exists():
        img = cv2.imread(str(path))
    else:
        img = (255 * np.ones((480, 640, 3), dtype=np.uint8)) if np is not None else None
    if img is None:
        return b""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return bytes(buf) if ok else b""

# ------------------------ Servers ------------------------

class AudioServer(threading.Thread):
    def __init__(self, host: str, port: int = AUDIO_PORT) -> None:
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self._stop = threading.Event()
        self._payload = self._load_payload()

    def _load_payload(self) -> bytes:
        """Load sim WAV, normalize to 16kHz mono 16-bit PCM, and return *raw PCM*."""
        if not SIM_WAV.exists():
            # Fall back to synthetic tone PCM
            return _make_pcm_bytes(3.0)
        try:
            with wave.open(str(SIM_WAV), "rb") as wf:
                nch = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                fr = wf.getframerate()
                nframes = wf.getnframes()
                raw = wf.readframes(nframes)

            # Ensure 16-bit samples
            if sampwidth != 2:
                raw = audioop.lin2lin(raw, sampwidth, 2)
                sampwidth = 2

            # Convert to mono if needed
            if nch == 2:
                raw = audioop.tomono(raw, 2, 0.5, 0.5)
                nch = 1
            elif nch != 1:
                # Fallback: first channel
                raw = audioop.tomono(raw, 2, 1.0, 0.0)
                nch = 1

            # Resample to 16kHz if needed
            target_sr = 16000
            if fr != target_sr:
                converted, _ = audioop.ratecv(raw, 2, 1, fr, target_sr, None)
                raw = converted
                fr = target_sr

            # At this point, `raw` is 16-bit mono 16kHz PCM
            return raw
        except Exception as e:
            print(f"[AUDIO] Failed to load sim.wav, using tone (reason: {e})")
            return _make_pcm_bytes(3.0)


    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(1)
            print(f"[AUDIO] Listening on {self.host}:{self.port}")
            while not self._stop.is_set():
                srv.settimeout(1.0)
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                with conn:
                    try:
                        # self._payload is raw PCM; length must be big-endian for `!I`
                        pcm = self._payload
                        header = struct.pack("!I", len(pcm))
                        conn.sendall(header)
                        conn.sendall(pcm)
                        time.sleep(1.0)
                    except Exception as e:
                        print(f"[AUDIO] Error: {e}")

    def stop(self):
        self._stop.set()


class VideoServer(threading.Thread):
    def __init__(self, host: str, port: int = VIDEO_PORT, fps: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.period = 1.0 / max(0.1, fps)
        self._stop = threading.Event()
        self._jpeg = _load_image_bytes()

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(1)
            print(f"[VIDEO] Listening on {self.host}:{self.port}")
            while not self._stop.is_set():
                srv.settimeout(1.0)
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                print(f"[VIDEO] Client {addr}")
                with conn:
                    try:
                        while not self._stop.is_set():
                            payload = self._jpeg
                            hdr = struct.pack(">I", len(payload))
                            conn.sendall(hdr)
                            conn.sendall(payload)
                            time.sleep(self.period)
                    except Exception as e:
                        print(f"[VIDEO] Error: {e}")

    def stop(self):
        self._stop.set()


class Mp3Sink(threading.Thread):
    def __init__(self, host: str, port: int = MP3_PORT) -> None:
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self._stop = threading.Event()

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(1)
            print(f"[MP3] Listening on {self.host}:{self.port}")
            while not self._stop.is_set():
                srv.settimeout(1.0)
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                print(f"[MP3] Client {addr}")
                total = 0
                with conn:
                    try:
                        while True:
                            data = conn.recv(4096)
                            if not data:
                                break
                            total += len(data)
                    except Exception as e:
                        print(f"[MP3] Error: {e}")
                print(f"[MP3] Received {total} bytes")

    def stop(self):
        self._stop.set()


class HapticSink(threading.Thread):
    def __init__(self, host: str, port: int = HAPTIC_PORT) -> None:
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self._stop = threading.Event()

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(1)
            print(f"[HAPTIC] Listening on {self.host}:{self.port}")
            while not self._stop.is_set():
                srv.settimeout(1.0)
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                print(f"[HAPTIC] Client {addr}")
                with conn:
                    try:
                        while True:
                            cmd = conn.recv(6)
                            if not cmd:
                                break
                            print(f"[HAPTIC] {cmd.decode(errors='ignore')}")
                    except Exception as e:
                        print(f"[HAPTIC] Error: {e}")

    def stop(self):
        self._stop.set()


def main():
    ap = argparse.ArgumentParser(description="ESP32 simulator for Smart Glass")
    ap.add_argument("--bind", default="127.0.0.1", help="Interface/IP to bind servers on")
    ap.add_argument("--fps", type=float, default=2.0, help="Video FPS")
    args = ap.parse_args()

    audio = AudioServer(args.bind)
    video = VideoServer(args.bind, fps=args.fps)
    mp3 = Mp3Sink(args.bind)
    haptic = HapticSink(args.bind)

    audio.start(); video.start(); mp3.start(); haptic.start()
    print("ESP32 simulator running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    print("Stopping...")
    audio.stop(); video.stop(); mp3.stop(); haptic.stop()

if __name__ == "__main__":
    main()
