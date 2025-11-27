# Smart Glass System Contract

A Jetson Nano base station and an ESP32-powered smart-glass headset collaborate to provide voice-activated navigation and scene-understanding assistance. The ESP32 captures sensor input and performs immediate actuation, while the Jetson aggregates perception, decision making, and response generation.

## Network Topology and Ports

| Link | Producer → Consumer | Port | Payload |
| --- | --- | --- | --- |
| Wake audio | ESP32 → Jetson | 1000 | PCM WAV stream (wake-word gated) |
| Video frames | ESP32 → Jetson | 2000 | MJPEG stream (length-prefixed JPEG) |
| Speech backchannel | Jetson → ESP32 | 3000 | MP3 audio (chunked) |
| Haptic commands | Jetson → ESP32 | 4000 | ASCII `LLLRRR` duty-cycle string |

The Jetson should bind as a TCP client to the ESP32 SoftAP for every channel.

## ESP32 Responsibilities

- **Wake Word + Audio Capture**
  - Run WakeNet on-device, trigger on “Hi ESP.”
  - Record up to `MAX_SPEECH_LENGTH` seconds and stream the raw WAV buffer to port `1000`.
- **Camera Streaming**
  - Acquire frames from the OV2640 camera.
  - Push each frame as a length-prefixed JPEG packet over port `2000`.
- **Speech Playback**
  - Listen for MP3 chunks on port `3000`.
  - Feed bytes into the decoder + I2S output for playback.
- **Vibration Motor Driver**
  - Accept fixed-width ASCII commands (`LLLRRR`) on port `4000`.
  - Map the received duty-cycles to the haptic actuators.

## Jetson Orchestration Blocks

Each software block must be composable so they can share a `TextBuffer` and run in a round-robin loop.

### 1. `receive_and_transcribe`

- Connect to port `1000`, respect a configurable timeout.
- If no data arrives before timeout, return `None`.
- Otherwise, pass the PCM stream to the Vosk recognizer and return the transcript string.

### 2. `classify_intent`

- Accept a transcript and produce `(intent_id, payload)` using `classify_intent.py`.
- Possible outcomes:
  - `0, None` – unclear request.
  - `1, destination` – navigation request; `destination` must be normalized for routing.
  - `2, prompt` – scene description request.

### 3. `obstacle_aware_navigation`

- Load the YOLO detector (`yolo11n.*`).
- Use Valhalla (or a stub planner) to compute and update routes.
- For each iteration:
  - Fuse GPS or localization updates.
  - If on path, enqueue turn-by-turn guidance into the `TextBuffer`.
  - If off path, enqueue a deviation warning and recompute.
  - Pull the latest frame from port `2000`, run obstacle detection, and send the resulting haptic command over port `4000` using the `LLLRRR` format.

### 4. `vlm_inference`

- Load the Moondream VLM from a fully local directory (default `moondream-2b-2025-04-14-4bit`) so everything runs offline; see `tests/test_moondream.py` for folder prep.
- Sample frames from port `2000` as needed.
- Generate a natural-language description (see `test_moondream.py`) and push it into the shared buffer.
- Immediately unload/clear GPU memory after each describe pass so the Jetson Nano stays within its VRAM budget.

### 5. `transmit_mp3`

- Monitor the shared buffer for queued text.
- Convert text to speech with the TTS module (see `test_tts.py`).
- Stream the MP3 response back to the ESP32 on port `3000` (`test_audio_transmit.py`).
- Keep this channel non-blocking so navigation and VLM inference stay responsive.

## End-to-End Event Loop

1. Wake phrase triggers audio capture on the headset.
2. Jetson receives and transcribes the user utterance.
3. Intent classification selects navigation or description workflows.
4. Navigation and VLM workers enqueue human-readable messages and haptic cues.
5. Speech synthesis streams responses back to the headset while haptics execute in parallel.

Every module in the workspace must honor these interfaces and ports to guarantee that the ESP32 and Jetson remain interoperable.

## Running the Jetson Pipeline

- Install Python dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- Some helpers rely on system packages:
  - `flite` and `aplay` for `tests/integration/test_tts.py`
  - NVIDIA Jetson-specific CUDA drivers/TensorRT for YOLO, VILA, and Moondream demos

- Ensure the Jetson has access to the required Python dependencies: `vosk`, `sentence-transformers`, `ultralytics`, `transformers`, `opencv-python`, and any GPU backends you plan to use (TensorRT/CUDA for YOLO and Moondream).
- Place the Moondream export in `moondream-2b-2025-04-14-4bit/` (default path expected by the pipeline) or pass `--moondream-dir` to point at another fully local folder that matches `tests/test_moondream.py`.
- Confirm the ESP32 soft AP is reachable (default `192.168.4.1`) and that the camera/audio services are already streaming on the documented ports.
- Launch the orchestrator:
  
  ```bash
  # With real ESP32 (replace host if different)
  python3 pipeline.py --esp-host 192.168.4.1 --loop --log INFO
  
  # With local simulator (see below)
  python3 pipeline.py --esp-host 127.0.0.1 --loop --log INFO
  ```

### Updated Orchestrator (`pipeline.py`)

The new `pipeline.py` implements the contract with lazy loading and memory-aware behavior:

CLI flags:

| Flag | Purpose |
|------|---------|
| `--esp-host` | ESP32 IP / simulator bind address for audio/video/vibro streams |
| `--audio-port`, `--video-port`, `--mp3-port`, `--vibro-port` | Override default ports 1000/2000/3000/4000 if your firmware differs |
| `--timeout` | Socket timeout (seconds) applied to every connection/read |
| `--yolo` | Explicit YOLO model path (TensorRT engine or `.pt`) |
| `--nav-host`, `--nav-port` | Valhalla (or stub) navigation IPC endpoint (default `127.0.0.1:5555`) |
| `--nav-timeout`, `--nav-costing`, `--nav-step-delay` | Fine-tune navigation service behavior/timeouts |
| `--moondream-dir` | Path to the fully local Moondream model folder |
| `--idle-sleep`, `--iter-sleep` | Sleep durations for idle vs. successful loop iterations |
| `--loop` | Continuous loop; otherwise run once |
| `--log` | Logging level |

Example (auto model selection + tuned backoff):

```bash
python3 pipeline.py --esp-host 127.0.0.1 --loop --idle-sleep 2.0 --iter-sleep 0.5 --log INFO
```

Memory notes:
- Moondream VLM is loaded only for description intents and immediately unloaded (`torch.cuda.empty_cache()`) to keep Jetson Nano memory pressure low.
- YOLO remains resident once loaded; if memory is tight you can restart the process between nav sessions or add an unload hook.

Fallback behavior:
- If Vosk returns no transcript, pipeline waits (`--idle-sleep`) and tries again.
- If YOLO cannot load any artifact in `YOLO/`, navigation still queues textual guidance and skips obstacle detection.
- If transformers/Moondream unavailable, description intents return a stub message.

### Local ESP32 Simulation

Run the built-in simulator to exercise the pipeline without hardware:

```bash
python3 esp32_simulator.py --bind 127.0.0.1
```

This starts TCP servers on 1000/2000/3000/4000. Then run the pipeline with `--esp-host 127.0.0.1` (and keep the default `--nav-host 127.0.0.1` for the nav stub).

Providing speech input:
- Place a test WAV at `audio/sim.wav` (any mono WAV). The simulator automatically normalizes it to 16 kHz mono 16‑bit PCM for Vosk.
- Quick generation with flite:
  ```bash
  flite -voice kal -t "navigate to the library" -o audio/sim.wav
  ```

### Troubleshooting

| Symptom | Resolution |
|---------|------------|
| Connection refused | Ensure simulator or ESP32 service is running; host matches `--esp-host`. |
| Vosk sampling mismatch (expected 16000, got 8000) | Simulator now resamples; restart simulator if you changed WAV. |
| "YOLO unavailable" | Verify ultralytics + dependencies installed (`pip show ultralytics`); ensure at least one model file exists in `YOLO/`. Install `torchvision` if import error mentions metadata. |
| Moondream out-of-memory | Description intents unload after each call; reduce frequency or remove VLM if still OOM. |
| No transcript | Confirm `audio/sim.wav` contains speech, not just silence/tone. |

### Minimal Once-Off Test

```bash
python3 pipeline.py --esp-host 127.0.0.1 --log DEBUG
```

Runs a single loop iteration: receives audio, classifies, performs nav or description, streams any speech response, then exits.

- Key CLI options:
  - `--esp-host`: ESP32 IP/hostname for audio/video/vibro streams.
  - `--nav-host` / `--nav-port`: Valhalla (or stub) navigation IPC endpoint.
  - `--timeout`: Socket timeout (seconds) applied to all channels.
  - `--yolo`: Path to the YOLO model artifact (TensorRT engine or `.pt`).
  - `--moondream-dir`: Path to the fully local Moondream export.
  - `--loop`: Keep the pipeline running continuously (omit for single iteration tests).
  - `--log`: Set logging granularity (e.g., `DEBUG`, `INFO`).

The runtime will auto-wire MJPEG frames, speech output, and motor commands. If any dependency is missing (e.g., YOLO, Moondream), the corresponding block automatically degrades to a safe no-op; you can supply mock components via dependency injection when running on a development machine.

## Testing Strategy

### Fast Unit Tests

Run the integration-focused unit tests that exercise the shared buffer, navigation/describer adapters, and speech output stubs:

```bash
cd /home/janchen/Documents/EECS473/Smart_Glass
python -m unittest tests.test_pipeline tests.test_runtime
```

These tests rely only on the lightweight stubs; no hardware or heavy ML models are required.

### Block-Level Smoke Tests

Each functional block has a dedicated script to validate its external dependencies in isolation:

- `receive_an_transcribe.py`: Connects to the audio stream on port `1000`, invokes Vosk, and prints the resulting transcript.
- `tests/test_vosk.py`: Offline transcription sanity check using local WAV files (no ESP32 required).
- `tests/test_audio_transmit.py`: Streams an MP3 file to the ESP32 speech port (`5000` in the test) to validate the return audio channel.
- `tests/test_moondream.py`: Loads the Moondream VLM and provides an interactive CLI for image prompts. Requires `transformers` and GPU support.
- `tests/test_vila.py`: Similar VLM validation using the VILA model (MLC backend).
- `tests/test_YOLO.py`: Runs YOLO inference on a sample frame; ensure `ultralytics` + TensorRT/CUDA are installed.
- `tests/test_tts.py`: Wrapper around `flite`/`aplay` for validating speech synthesis locally.

Optional Jetson-specific dependencies that are not available on PyPI:

- `nano-llm`: install from source (`git clone https://github.com/mlc-ai/nano-llm && pip install -e .`).
- `jetson-utils`: use NVIDIA's prebuilt wheels or build from source following the [jetson-utils guide](https://github.com/dusty-nv/jetson-utils).

> Tip: Execute these scripts individually after installing their prerequisites to confirm each dependency is functional before running the full pipeline.

### Next Enhancements (Optional)
- Add proper route planner integration (Valhalla stub replacement).
- Maintain a short-lived Moondream cache with idle timeout instead of per-intent unload for faster consecutive descriptions.
- Stream real TTS synthesis as MP3 instead of canned sample.
- Add unit tests for the new pipeline adapters.