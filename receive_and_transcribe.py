"""Audio reception + transcription block used by the Smart Glass pipeline."""

from __future__ import annotations

import io
import json
import socket
import struct
import sys
import wave
from dataclasses import dataclass
from typing import Optional
import time

from vosk import KaldiRecognizer, Model, SetLogLevel

SetLogLevel(-1)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 1000
DEFAULT_TIMEOUT = 10.0
DEFAULT_MODEL = "vosk-model-small-en-us-0.15"
CHUNK = 4096

PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_SAMPWIDTH = 2

_MODEL: Optional[Model] = None


class AudioStreamError(RuntimeError):
    """Raised for recoverable errors during streaming."""


def load_vosk_model(model: str = DEFAULT_MODEL) -> Model:
    """Load and cache the Vosk model for the requested language."""

    global _MODEL
    if _MODEL is None:
        _MODEL = Model(model_name=model)
    return _MODEL


def read_exact(sock: socket.socket, size: int) -> bytes:
    """Read exactly *size* bytes from *sock* or raise an error."""

    buf = bytearray()
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise AudioStreamError("Socket closed while reading header")
        buf.extend(chunk)
    return bytes(buf)


def pcm_to_wav_bytes(
    pcm_data: bytes,
    sample_rate: int = PCM_SAMPLE_RATE,
    channels: int = PCM_CHANNELS,
    sampwidth: int = PCM_SAMPWIDTH,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buffer.getvalue()


def _extract_text(blob: str) -> str:
    try:
        result = json.loads(blob)
    except json.JSONDecodeError:
        return ""
    return result.get("text", "").strip()


@dataclass
class TranscriptionResult:
    text: Optional[str]
    sample_rate: int
    channels: int


def receive_and_transcribe(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    model_name: str = DEFAULT_MODEL
) -> Optional[str]:
    """Connect to the ESP32 stream, convert it to WAV, transcribe, and return text."""

    model = load_vosk_model(model=model_name)
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            length_raw = read_exact(sock, 4)
            (pcm_length,) = struct.unpack("!I", length_raw)
            if pcm_length <= 0:
                raise AudioStreamError(f"Invalid PCM length: {pcm_length}")
            pcm_data = read_exact(sock, pcm_length)

        wav_bytes = pcm_to_wav_bytes(pcm_data)
        return _transcribe_wav_bytes(model, wav_bytes)
    except (OSError, AudioStreamError, ValueError) as exc:
        print(f"[{time.time()}] Waiting for audio from ESP ...")
        return None


def _transcribe_wav_bytes(model: Model, wav_bytes: bytes) -> Optional[str]:
    buffer = io.BytesIO(wav_bytes)
    with wave.open(buffer, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()

        if sampwidth != PCM_SAMPWIDTH:
            raise ValueError(f"Unsupported sample width {sampwidth}; expected {PCM_SAMPWIDTH}")

        recognizer = KaldiRecognizer(model, sample_rate)
        recognizer.SetWords(True)

        bytes_per_frame = channels * sampwidth
        frames_per_chunk = max(1, CHUNK // max(1, bytes_per_frame))
        transcript_parts = []

        while True:
            chunk = wf.readframes(frames_per_chunk)
            if not chunk:
                break
            if recognizer.AcceptWaveform(chunk):
                text = _extract_text(recognizer.Result())
                if text:
                    transcript_parts.append(text)

    final_text = _extract_text(recognizer.FinalResult())
    if final_text:
        transcript_parts.append(final_text)

    transcript = " ".join(transcript_parts).strip()
    return transcript or None

def main() -> None:
    print(f"[INFO] Listening for audio on {DEFAULT_HOST}:{DEFAULT_PORT} ...", flush=True)
    transcript = receive_and_transcribe()
    if transcript:
        print(f"[FINAL] {transcript}")
    else:
        print("[INFO] No transcript produced.")


if __name__ == "__main__":
    main()
