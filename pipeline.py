#!/usr/bin/env python3
"""Smart Glass minimal runtime pipeline.

Implements the contract from README using existing helper scripts:
  1. Receive + transcribe audio (port 1000) via receive_and_transcribe.receive_and_transcribe
  2. Classify intent via classify_intent.classify
  3. Navigation worker (YOLO + haptics) for intent 1
  4. Scene description worker (Moondream VLM) for intent 2 with aggressive GPU memory release
  5. Speech output: text buffered -> (optional) MP3 stream (port 3000) and console log

All heavy modules are loaded lazily and released ASAP to preserve Jetson Nano memory.
If a dependency is missing (ultralytics / transformers / flite), safe stubs are used.

CLI:
  python3 pipeline.py --esp-host 192.168.4.1 --loop --nav-host 127.0.0.1 --log INFO
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import shutil
import socket
import struct
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("smart_glass.pipeline")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

DEFAULT_MOONDREAM_DIR = Path("/home/team15/Documents/SmartGlass/moondream-2b-2025-04-14-4bit").resolve()
TEMP_SPEECH_DIR = (Path(__file__).resolve().parent / "temp").resolve()
TEMP_SPEECH_DIR.mkdir(parents=True, exist_ok=True)

# from jetson_utils import loadImage, cudaFromNumpy
# _HAS_JETSON_UTILS = True

from tests.test_tts import flite_tts, piper_tts  # type: ignore
from tests.test_audio_transmit import convert_sample_rate  # type: ignore
from pydub import AudioSegment  # type: ignore
from tests import test_moondream as moondream_api  # type: ignore

# ---------------------------------------------------------------------------
# Imports from existing modules (best-effort)
# ---------------------------------------------------------------------------
from receive_and_transcribe import receive_and_transcribe  # type: ignore

from classify_intent_failed import classify as classify_intent, get_judge as get_intent_judge  # type: ignore

from destinations import safe_destination

#Color Logging
class ColorFormatter(logging.Formatter):
    # ANSI escape codes
    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG:    "\033[36m",  # cyan
        logging.INFO:     "\033[32m",  # green
        logging.WARNING:  "\033[33m",  # yellow
        logging.ERROR:    "\033[31m",  # red
        logging.CRITICAL: "\033[41m",  # red background
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        reset = self.RESET if color else ""
        return f"{color}{msg}{reset}"


# ---------------------------------------------------------------------------
# Shared text buffer
# ---------------------------------------------------------------------------
class TextBuffer:
    def __init__(self) -> None:
        self._items: List[str] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def push(self, msg: str) -> None:
        msg = msg.strip()
        if msg:
            logger.debug("Buffer push: %s", msg)
            with self._not_empty:
                self._items.append(msg)
                self._not_empty.notify()
    
    def pop_one(self, timeout: float | None = None):
        with self._not_empty:
            if not self._items:
                if timeout is None:
                    self._not_empty.wait()
                else:
                    if not self._not_empty.wait(timeout=timeout):
                        return None
            if not self._items:
                return None
            return self._items.pop(0)

    def drain(self) -> List[str]:
        with self._lock:
            out = self._items[:]
            self._items.clear()
        return out

# ---------------------------------------------------------------------------
# Utilities for MJPEG + vibro protocol
# ---------------------------------------------------------------------------

def _read_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        pkt = sock.recv(n - len(buf))
        if not pkt:
            return None
        buf.extend(pkt)
    return bytes(buf)


def _recv_mjpeg_frame(sock: socket.socket):
    try:
        import cv2, numpy as np  # type: ignore
    except Exception:
        return None
    hdr = _read_exact(sock, 4)
    if not hdr:
        return None
    (length,) = struct.unpack(">I", hdr)
    payload = _read_exact(sock, length)
    if not payload:
        return None
    # Save the raw JPEG payload to disk for VLM debugging/inspection.
    try:
        from pathlib import Path
        # timestamp format: MMDDHHMMSS (month-day-hour-minute-second)
        timestamp = time.strftime("%m%d%H%M%S", time.localtime())
        out_dir = Path("VLM_image")
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / f"vlm_{timestamp}.jpep"
        with file_path.open("wb") as f:
            f.write(payload)
    except Exception as exc:
        logger.debug("[VLM] Failed to save MJPEG frame to disk: %s", exc)

    img = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def _prepare_moondream_env(model_dir: Path) -> bool:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.exists():
        logger.warning("Moondream directory not found: %s", model_dir)
        return False
    cache_root = (model_dir.parent / ".hf_cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    moondream_api.MODEL_DIR = model_dir
    moondream_api.CACHE_ROOT = cache_root
    required = ["config.json", "model.safetensors", "hf_moondream.py", "tokenizer.json"]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"[VLM] Missing required files in {model_dir}: {missing}")
    return True

# ---------------------------------------------------------------------------
# Navigation block (YOLO + haptics)
# ---------------------------------------------------------------------------
class NavigationWorker:
    """Navigation worker with OSM-Valhalla routing, YOLO obstacle detection and haptic feedback."""
    def __init__(
        self,
        esp_host: str,
        video_port: int,
        vibro_port: int,
        model_path: str,
        *,
        nav_host: Optional[str] = None,
        nav_port: Optional[int] = None,
        nav_timeout: float = 5.0,
        nav_costing: str = "pedestrian",
        nav_step_delay: float = 0.1,
    ) -> None:
        self.esp_host = esp_host
        self.video_port = video_port
        self.vibro_port = vibro_port
        self.model_path = model_path
        self.nav_host = nav_host
        self.nav_port = nav_port
        self.nav_timeout = nav_timeout
        self.nav_costing = nav_costing
        self.nav_step_delay = nav_step_delay
        self._yolo_model = None
        self._nav_thread: Optional[threading.Thread] = None
        self._nav_stop = threading.Event()
        self._nav_sock: Optional[socket.socket] = None
        self._read_buf = b""
        try:
            from ultralytics import YOLO  # type: ignore
            self._YOLO = YOLO
        except Exception:
            self._YOLO = None

    def _ensure_model(self):
        """Load YOLO model if not already loaded."""
        if self._yolo_model or not self._YOLO:
            return
        try:
            self._yolo_model = self._YOLO(self.model_path)
            logger.info("[Nav] YOLO model loaded: %s", self.model_path)
        except Exception as exc:
            logger.error(f"[Nav] YOLO model load Failed :{exc}.", exc)

            # raise RuntimeError(f"[Nav] YOLO model load failed: {exc}") from exc

    def _area_to_speed(self, area_frac: float) -> int:
        """Convert bounding box area fraction to vibro speed (0-100)."""
        area_frac = max(0.0, min(1.0, area_frac))
        return int(round((area_frac ** 0.8) * 100))

    def _derive_haptics(self, result, shape: Tuple[int, int, int]):
        """Derive left/right vibro speeds and summary from YOLO result."""
        import numpy as np  # type: ignore
        H, W = shape[:2]
        frame_area = float(H * W)
        left = right = 0
        summary: List[str] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return left, right, summary
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.int().cpu().numpy()
        names = result.names
        for (x1, y1, x2, y2), cid in zip(xyxy, cls_ids):
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area_frac = (w * h) / frame_area
            speed = self._area_to_speed(area_frac)
            cx = (x1 + x2) / 2.0
            region = cx / W
            name = names.get(int(cid), str(cid)).lower()
            if region < 1/3:
                left = max(left, speed)
                summary.append(f"{name}@{speed}%←")
            elif region > 2/3:
                right = max(right, speed)
                summary.append(f"{name}@{speed}%→")
            else:
                scaled = int(round(speed * 0.75))
                left = max(left, scaled)
                right = max(right, scaled)
                summary.append(f"{name}@{scaled}%↔")
        return left, right, summary

    def _send_vibro(self, left: int, right: int) -> None:
        """Send vibro command to ESP32."""
        msg = f"{max(0,min(left,999)):03d}{max(0,min(right,999)):03d}".encode("ascii")
        try:
            with socket.create_connection((self.esp_host, self.vibro_port), timeout=5.0) as s:
                s.sendall(msg)
        except Exception:
            logger.debug("Vibro send failed")

    def run_once(self, destination: str, buffer: TextBuffer) -> None:
        buffer.push(f"Trying navigation to {destination}.")
        if not self._start_nav_stream(destination, buffer):
            buffer.push(f"Stay on course towards {destination}.")
        self._run_obstacle_detection(buffer)

    def _start_nav_stream(self, destination: str, buffer: TextBuffer) -> bool:
        """Start navigation instruction streaming thread."""
        if not self.nav_host or not self.nav_port:
            raise RuntimeError("[Nav] Navigation host/port not configured")
        self._stop_nav_stream()
        stop_event = threading.Event()
        self._nav_stop = stop_event
        thread = threading.Thread(
            target=self._stream_nav_instructions,
            args=(destination, buffer, stop_event),
            daemon=True,
        )
        self._nav_thread = thread
        thread.start()
        return True

    def _stop_nav_stream(self) -> None:
        """Stop navigation instruction streaming thread."""
        logger.info("Stop streaming navigation instructions...")
        if self._nav_thread and self._nav_thread.is_alive():
            self._nav_stop.set()
            if self._nav_sock:
                try:
                    self._nav_sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    self._nav_sock.close()
                except Exception:
                    pass
                finally:
                    self._nav_sock = None
            # self._nav_thread.join(timeout=1.0)
        self._nav_thread = None
        self._nav_stop = threading.Event()

    def _readline(self, sock_obj: socket.socket) -> str:
        """Read a line (ending with \n) from socket."""
        while b"\n" not in self._read_buf:
            if self._nav_stop.is_set():
                break
            try:
                chunk = sock_obj.recv(4096)
            except socket.timeout:
                logger.warning("[Nav] Reading navigation port failed: timeout")
                break
            if not chunk:
                logger.warning("[Nav] Reading navigation port failed: None")
                break
            self._read_buf += chunk
        if b"\n" not in self._read_buf:
            line, self._read_buf = self._read_buf, b""
        else:
            line, self._read_buf = self._read_buf.split(b"\n", 1)
        return line.decode("utf-8").strip() if line else ""

    def _stream_nav_instructions(self, destination: str, buffer: TextBuffer, stop_event: threading.Event) -> None:
        """Stream navigation instructions from Valhalla navigation service."""
        payload = {
            "destination": destination,
            "costing": self.nav_costing,
            "step_delay": self.nav_step_delay,
            "poll_delay": self.nav_step_delay,
        }
        try:
            with socket.create_connection((self.nav_host, self.nav_port), timeout=self.nav_timeout) as sock_obj:
                self._nav_sock = sock_obj
                sock_obj.settimeout(self.nav_timeout)
                sock_obj.sendall(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
                while not stop_event.is_set():
                    logger.info("Streaming navigation instructions...")
                    line = self._readline(sock_obj)
                    if not line:
                        continue
                    try:
                        message = json.loads(line)
                        event = message.get("event")
                        text = message.get("text")
                        if event == "instruction" and text:
                            logger.info(f"[Nav] {event.capitalize()}: {text}")
                            buffer.push(text)
                        if event == "error":
                            logger.error(f"[Nav] Navigation error: {text}")
                            buffer.push(text)
                        if event == "done":
                            break
                    except json.JSONDecodeError:
                        logger.warning("[Nav] JSON decode failed for line: %r", line)
                        #buffer.push(line)
        except Exception as exc:
            logger.error(f"[Nav] Navigation stream failed: {exc}.", exc)
            # raise RuntimeError(f"[Nav] Navigation stream failed: {exc}") from exc
        finally:
            self._nav_sock = None
            if threading.current_thread() is self._nav_thread:
                self._nav_thread = None

    def _run_obstacle_detection(self, buffer: TextBuffer) -> None:
        """Run obstacle detection and haptic feedback loop once."""
        if not self._YOLO:
            logger.error(f"[Nav] Ultralytics YOLO not available.")
            return
            # raise RuntimeError("[Nav] Ultralytics YOLO not available")
        self._ensure_model()
        if not self._yolo_model:
            logger.error("[Nav] YOLO model load failed.")
            return
            # raise RuntimeError("[Nav] YOLO model load failed")
        try:
            with socket.create_connection((self.esp_host, self.video_port), timeout=5.0) as vs:
                frame = _recv_mjpeg_frame(vs)
            if frame is None:
                logger.debug("No navigation frame received")
                return
            results = self._yolo_model(frame, conf=0.25, iou=0.45, verbose=False)
            result = results[0]
            left, right, summary = self._derive_haptics(result, frame.shape)
            # if summary:
            #     buffer.push("Obstacles: " + ", ".join(summary))
            self._send_vibro(left, right)
        except Exception as exc:
            logger.error(f"[Nav] Obstacle detection failed: {exc}", exc)
            # raise RuntimeError(f"[Nav] Obstacle detection failed: {exc}") from exc

    def wait_for_stream(self, timeout: Optional[float] = None) -> None:
        thread = self._nav_thread
        if thread and thread.is_alive():
            thread.join(timeout)

# ---------------------------------------------------------------------------
# Scene description block (Moondream VLM) with unload
# ---------------------------------------------------------------------------
class VLMDescriber:
    def __init__(
        self,
        host: str,
        video_port: int,
        unload: bool = False,
        model_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.host = host
        self.video_port = video_port
        self.unload = unload
        self.model_dir = Path(model_dir).expanduser() if model_dir else DEFAULT_MOONDREAM_DIR
        self._env_ready = _prepare_moondream_env(self.model_dir)
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model

        if not self._env_ready:
            self._env_ready = _prepare_moondream_env(self.model_dir)
        try:
            model = moondream_api.get_model()  # type: ignore[union-attr]
            self._model = model
            logger.info("[VLM] Moondream model loaded successfully.")
            return model
        except Exception as exc:
            raise RuntimeError(f"[VLM] Moondream model load failed: {exc}") from exc

    def _free(self):
        if not self._model:
            return
        try:
            import torch  # type: ignore
            self._model = None
            if moondream_api:
                moondream_api._MODEL = None  # type: ignore[attr-defined]
            torch.cuda.empty_cache()
            logger.info("[VLM] Moondream model unloaded; GPU cache cleared")
        except Exception:
            pass

    def describe_once(self, prompt: str, buffer: TextBuffer) -> None:
        # Grab one frame (optional)
        frame_np = None
        try:
            with socket.create_connection((self.host, self.video_port), timeout=5.0) as vs:
                frame_np = _recv_mjpeg_frame(vs)
        except Exception:
            logger.warning("[VLM] Frame receive failed; proceeding without image")
        try:
            model = self._load()
        except Exception as exc:
            logger.error("[VLM] Cannot load model for description: %s", exc)
            buffer.push("Sorry, description model is unavailable, please reboot the machine.")
            return
        
        if not model:
            buffer.push("Sorry, description model is unavailable, please reboot the machine.")
            return
    
        # Convert to PIL if possible
        pil_img = None
        if frame_np is not None:
            try:
                import cv2  # type: ignore
                from PIL import Image  # type: ignore
                rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
            except Exception:
                pil_img = None
        try:
            image_obj = pil_img if pil_img is not None else frame_np
            if moondream_api and image_obj is not None:
                text = moondream_api.describe_image(image_obj, prompt)  # type: ignore[union-attr]
            else:
                out = model.query(image_obj, prompt)
                text = out.get("answer", "").strip() if isinstance(out, dict) else str(out)
            buffer.push((text or f"(No answer) {prompt}").strip())
        except Exception as exc:
           logger.warning("[VLM] Description failed: %s", exc)
        finally:
            if self.unload:
                self._free()

# ---------------------------------------------------------------------------
# Speech output (console + optional MP3 stream)
# ---------------------------------------------------------------------------
class SpeechOutput:
    def __init__(self, host: str, port: int, buffer: TextBuffer) -> None:
        self.host = host
        self.port = port
        self.output_dir = TEMP_SPEECH_DIR
        self.buffer = buffer

        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            msg = self.buffer.pop_one(timeout=0.1)
            if msg is None:
                continue

            try:
                logger.info("[SpeechOutput] Speaking: %s", msg)
                self.flush([msg])
            except Exception as exc:
                logger.error("[SpeechOutput] Failed to speak: %s", exc)      
    
    def stop(self) -> None:
        """请求停止后台线程（可在程序退出前调用）。"""
        self._stop.set()
        try:
            self._worker.join(timeout=1.0)
        except Exception:
            pass

    def flush(self, messages: Sequence[str]) -> None:
        if not messages:
            logger.info("[TTS] No message to convert to speech.")
            return
        payload = "\n".join(messages)
        logger.info("[SPEECH]\n%s", payload)
        try:
            with tempfile.TemporaryDirectory(prefix="speech-out-") as tmpdir:
                prepared = self._prepare_mp3(payload, Path(tmpdir))
                if not prepared:
                    return
                self._stream_mp3(prepared)
        except Exception as exc:
            raise RuntimeError(f"[TTS] Speech output failed: {exc}") from exc

    def _prepare_mp3(self, text: str, tmp_dir: Path) -> Optional[Path]:
        wav_path = tmp_dir / "speech.wav"
        try:
            #flite_tts(text, str(wav_path))
            piper_tts(text, str(wav_path))
        except SystemExit as exc:
            raise RuntimeError("[TTS] flite_tts exited unexpectedly") from exc
        except Exception as exc:
            raise RuntimeError(f"[TTS] flite_tts failed: {exc}") from exc
        mp3_path = tmp_dir / "speech.mp3"
        try:
            audio = AudioSegment.from_wav(str(wav_path))
            audio.export(str(mp3_path), format="mp3", bitrate="256k")
        except Exception as exc:
            raise RuntimeError(f"[TTS] MP3 conversion failed: {exc}") from exc
        try:
            processed = convert_sample_rate(str(mp3_path))
        except Exception as exc:
            raise RuntimeError(f"[TTS] MP3 sample rate conversion failed: {exc}") from exc
        processed_path = Path(processed)
        timestamp = int(time.time() * 1000)
        dest_path = self.output_dir / f"speech-{timestamp}.mp3"
        try:
            shutil.copy2(processed_path, dest_path)
        except Exception as exc:
            raise RuntimeError(f"[TTS] Failed to store speech output: {exc}") from exc
        return dest_path

    def _stream_mp3(self, mp3_path: Path) -> None:
        try:
            with socket.create_connection((self.host, self.port), timeout=5.0) as s, mp3_path.open("rb") as f:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    s.sendall(chunk)
            logger.debug("MP3 stream sent (%s)", mp3_path.name)
        except Exception as exc:
            logger.debug("MP3 stream skipped: %s", exc)
    
    def play_file(self, mp3_path: Union[str, Path]) -> None:
        """Play an existing MP3 file over the MP3 TCP stream, after
        converting it to 16 kHz, 16-bit, mono using convert_sample_rate().
        """
        path = Path(mp3_path)
        if not path.is_file():
            logger.error("[TTS] Startup MP3 file not found: %s", path)
            return

        try:
            # This should produce a 16 kHz, 16-bit, mono MP3 (same as in _prepare_mp3)
            processed = convert_sample_rate(str(path))
            processed_path = Path(processed)
        except Exception as exc:
            logger.error("[TTS] Failed to convert startup MP3 to 16kHz mono: %s", exc)
            return

        logger.info("[TTS] Playing startup MP3 (16kHz mono): %s", processed_path)
        self._stream_mp3(processed_path)

# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
@dataclass
class SmartGlassPipeline:
    esp_host: str
    audio_port: int
    video_port: int
    mp3_port: int
    vibro_port: int
    timeout: float
    yolo_model: str
    nav_host: Optional[str] = None
    nav_port: Optional[int] = None
    nav_timeout: float = 5.0
    nav_costing: str = "pedestrian"
    nav_step_delay: float = 0.1
    moondream_dir: Optional[Union[str, Path]] = None
    buffer: TextBuffer = field(default_factory=TextBuffer)
    idle_sleep: float = 1.0   # sleep when no transcript
    iter_sleep: float = 0.2   # sleep after a processed iteration

    def __post_init__(self):
        self.nav = NavigationWorker(
            self.esp_host,
            self.video_port,
            self.vibro_port,
            self.yolo_model,
            nav_host=self.nav_host,
            nav_port=self.nav_port,
            nav_timeout=self.nav_timeout,
            nav_costing=self.nav_costing,
            nav_step_delay=self.nav_step_delay,
        )
        self.vlm = VLMDescriber(
            self.esp_host,
            self.video_port,
            unload=False,
            model_dir=self.moondream_dir,
        )
        self.speaker = SpeechOutput(self.esp_host, self.mp3_port, self.buffer)
        self.intent_judge = get_intent_judge()

    def _poll_nav_port_message(self) -> None:
        if not self.nav_host or not self.nav_port:
            return
        # Avoid colliding with the active navigation stream
        if self.nav._nav_thread and self.nav._nav_thread.is_alive():
            return
        try:
            with socket.create_connection((self.nav_host, self.nav_port), timeout=0.2) as sock:
                sock.settimeout(0.1)
                try:
                    data = sock.recv(4096)
                except socket.timeout:
                    return
                if not data:
                    return
                decoded = data.decode("utf-8", errors="ignore")
                for line in decoded.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        text = payload.get("text") if isinstance(payload, dict) else None
                        if text:
                            self.buffer.push(text)
                        else:
                            self.buffer.push(line)
                    except json.JSONDecodeError:
                        self.buffer.push(line)
        except (ConnectionRefusedError, TimeoutError, socket.timeout):
            logger.debug("[Pipeline] Nav port poll unavailable")
        except Exception as exc:
            logger.debug("[Pipeline] Nav port poll skipped: %s", exc)

    def _receive_transcript(self) -> Optional[str]:
        if not receive_and_transcribe:
            return None
        return receive_and_transcribe(host=self.esp_host, port=self.audio_port, timeout=self.timeout)

    def process_once(self) -> bool:
        self._flush()
        transcript = self._receive_transcript()
        if not transcript:
            logger.debug("[Pipeline] No transcript available; flushing any queued text")
            self._flush()
            return False
        logger.info("[Pipeline] Transcript: %s", transcript)
        intent, payload = classify_intent(transcript, judge=self.intent_judge)
        logger.info("[Pipeline] Intent=%s payload=%s", intent, payload)
        if intent == 1:
            destination = safe_destination(payload)   
            if destination:
                self.nav.run_once(destination, self.buffer)
            else:
                self.buffer.push("Navigation intent detected but no destination provided.")
        elif intent == 2:
            self.vlm.describe_once(payload or transcript, self.buffer)
        else:
            self.buffer.push(f"I am not sure how to help with '{transcript}'.")
        self._flush()
        return True

    def _flush(self):
        self._poll_nav_port_message()


    def run_loop(self):  # interactive loop
        logger.info("[Pipeline] Starting Smart Glass pipeline loop")
        try:
            while True:
                worked = self.process_once()
                time.sleep(self.iter_sleep if worked else self.idle_sleep)
        except KeyboardInterrupt:
            logger.info("[Pipeline] Ctrl+C received, shutting down Smart Glass pipeline...")
        finally:
            # Stop any ongoing navigation stream
            try:
                self.nav._stop_nav_stream()
            except Exception as exc:
                logger.debug("[Pipeline] Error while stopping nav stream during shutdown: %s", exc)

            # *** IMPORTANT PART: unload VLM model from GPU memory ***
            try:
                self.vlm._free()
            except Exception as exc:
                logger.debug("[Pipeline] Error while unloading VLM model: %s", exc)


# ---------------------------------------------------------------------------
# Factory & CLI
# ---------------------------------------------------------------------------

def build_pipeline(**kwargs) -> SmartGlassPipeline:
    return SmartGlassPipeline(**kwargs)


def main():
    ap = argparse.ArgumentParser(description="Minimal Smart Glass runtime")
    ap.add_argument("--esp-host", default="192.168.4.1")
    ap.add_argument("--audio-port", type=int, default=1000)
    ap.add_argument("--video-port", type=int, default=2000)
    ap.add_argument("--mp3-port", type=int, default=3000)
    ap.add_argument("--vibro-port", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--yolo", default="YOLO/yolo11n.engine")
    ap.add_argument("--nav-host", default="127.0.0.1", help="Navigation IPC host")
    ap.add_argument("--nav-port", type=int, default=5555, help="Navigation IPC port")
    ap.add_argument("--nav-timeout", type=float, default=5.0, help="Navigation IPC socket timeout")
    ap.add_argument("--nav-costing", default="pedestrian", choices=["auto","pedestrian","bicycle"], help="Valhalla costing sent to navigation service")
    ap.add_argument("--nav-step-delay", type=float, default=0.1, help="Step delay hint (seconds) for simulated navigation service")
    ap.add_argument("--moondream-dir", default=str(DEFAULT_MOONDREAM_DIR), help="Path to local Moondream model directory")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--idle-sleep", type=float, default=1.0, help="Sleep seconds when no transcript")
    ap.add_argument("--iter-sleep", type=float, default=0.2, help="Sleep seconds after a processed iteration")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    # Color logging
    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        handlers=[handler],
    )

    pipeline = build_pipeline(esp_host=args.esp_host,
                              audio_port=args.audio_port,
                              video_port=args.video_port,
                              mp3_port=args.mp3_port,
                              vibro_port=args.vibro_port,
                              timeout=args.timeout,
                              yolo_model=args.yolo,
                              nav_host=args.nav_host,
                              nav_port=args.nav_port,
                              nav_timeout=args.nav_timeout,
                              nav_costing=args.nav_costing,
                              nav_step_delay=args.nav_step_delay,
                              moondream_dir=args.moondream_dir)
    pipeline.idle_sleep = args.idle_sleep
    pipeline.iter_sleep = args.iter_sleep

    # Play Startup music
    music_path = "hajimi.mp3"
    try:
        pipeline.speaker.play_file(music_path)
    except Exception as exc:
        logger.error("[Startup] Failed to play startup sound: %s", exc)

    # Preload VLM model
    vlm_ready = False
    try:
        pipeline.vlm._load()
        logger.info("[VLM] Preloading Moondream model ...")
        vlm_ready = True
    except Exception as exc:
        logger.error("[VLM] Failed to preload Moondream model: %s", exc)
        pipeline.buffer.push("Scene description model failed to load.")
    if vlm_ready:
        pipeline.buffer.push("Welcome to Smart Glass! I am Hajimi!")
        pipeline._flush()

    if args.loop:
        pipeline.run_loop()
    else:
        pipeline.process_once()
        pipeline.nav.wait_for_stream(timeout=args.nav_timeout)
        pipeline._flush()

if __name__ == "__main__":
    main()
