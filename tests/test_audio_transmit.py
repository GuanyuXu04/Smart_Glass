import os
import socket
from pathlib import Path
from time import perf_counter
from pydub import AudioSegment
from pathlib import Path
from typing import Optional
from pydub import AudioSegment

ESP_IP   = "192.168.4.1"   # ESP32 SoftAP default gateway
ESP_PORT = 3000           # Must match ESP32 TCP server port
FILE     = "audio/tts.mp3"
CHUNK    = 4096
TIMEOUT  = 10              # seconds

def convert_sample_rate(src_path: str, max_size: int = 1048575) -> Path:
    """
    Convert the given MP3 file to 16 kHz mono using pydub (ffmpeg),
    and clamp the output so that its file size does not exceed max_size bytes.

    If max_size is None, it defaults to ~30 seconds of audio at 256 kbps.

    Returns the path of the converted file (same directory, suffix "_16k").
    """
    src = Path(src_path)
    if not src.is_file():
        raise FileNotFoundError(f"Input file not found: {src}")

    dst = src.with_stem(src.stem + "_16k").with_suffix(".mp3")

    print(f"[INFO] Converting {src.name} → 16 kHz mono ...")
    audio = AudioSegment.from_file(src)
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Target bitrate for export (256 kbps)
    target_bitrate_bps = 256_000

    # Default: at most 30 seconds of audio, converted to bytes for clamping
    if max_size is None:
        max_duration_sec_default = 30.0
        max_size = int((max_duration_sec_default * target_bitrate_bps) / 8)

    if max_size > 0:
        # Max duration in seconds given the file size limit and bitrate
        max_duration_sec = (max_size * 8) / target_bitrate_bps
        max_duration_ms = int(max_duration_sec * 1000)

        # Clamp audio duration to fit the estimated max_size
        if len(audio) > max_duration_ms:
            print(
                f"[INFO] Clamping audio from {len(audio)} ms to {max_duration_ms} ms "
                f"(≈ {max_duration_sec:.2f} s) to fit in ~{max_size} bytes "
                f"at {target_bitrate_bps // 1000} kbps."
            )
            audio = audio[:max_duration_ms]

    audio.export(dst, format="mp3", bitrate="256k")
    print(f"[OK] Converted file written to: {dst.name}")
    return dst


def main():
    converted = convert_sample_rate(FILE)
    p = Path(converted)
    if not p.is_file():
        print(f"[ERR] File not found: {p.resolve()}")
        return 1

    total = p.stat().st_size
    print(f"[INFO] Connecting to {ESP_IP}:{ESP_PORT} ...")
    try:
        with socket.create_connection((ESP_IP, ESP_PORT), timeout=TIMEOUT) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sent = 0
            t0 = perf_counter()
            with p.open("rb") as f:
                while True:
                    data = f.read(CHUNK)
                    if not data:
                        break
                    s.sendall(data)
                    sent += len(data)
                    # lightweight progress
                    if total > 0:
                        pct = (sent * 100) // total
                        print(f"\r[TX] {sent}/{total} bytes ({pct}%)", end="", flush=True)
            # optional: half-close to signal EOF cleanly
            try:
                s.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            dt = perf_counter() - t0
            print()  # newline after progress
            mb = sent / (1024*1024)
            rate = (mb / dt) if dt > 0 else 0.0
            print(f"[OK] Sent {sent} bytes in {dt:.2f}s ({rate:.2f} MB/s)")
            return 0
    except (OSError, socket.timeout) as e:
        print(f"[ERR] Socket error: {e}")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
