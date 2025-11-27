import time
import wave
import json
from typing import Optional

try:
    from vosk import KaldiRecognizer, Model, SetLogLevel  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    KaldiRecognizer = None  # type: ignore
    Model = None  # type: ignore
    SetLogLevel = lambda *_args, **_kwargs: None  # type: ignore

# Silence Vosk logs (set to 0 for normal, -1 to disable)
if SetLogLevel is not None:
    SetLogLevel(0)

# Cache the model so repeated calls don't reload it
_VOSK_MODEL: Optional[Model] = None

def _get_model() -> Model:
    global _VOSK_MODEL
    if Model is None:
        raise RuntimeError("vosk is not installed; transcription demo unavailable.")
    if _VOSK_MODEL is None:
        # Same model init as your starter code
        _VOSK_MODEL = Model(lang="en-us")
        # Or: _VOSK_MODEL = Model(model_name="vosk-model-en-us-0.21")
        # Or: _VOSK_MODEL = Model("models/en")
    return _VOSK_MODEL

def speech_to_text(wav_path: str) -> str:
    """
    Transcribe a WAV file (mono, 16-bit PCM) to text using Vosk.
    Returns the recognized text as a single string.

    :param wav_path: Path to a correctly formatted WAV file.
    :raises ValueError: if the WAV is not mono/16-bit PCM.
    :raises FileNotFoundError: if the file doesn't exist.
    """
    # Open and validate the WAV
    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise ValueError("Audio file must be WAV format mono 16-bit PCM (uncompressed).")

        model = _get_model()
        if KaldiRecognizer is None:
            raise RuntimeError("vosk is not installed; cannot create recognizer.")
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        rec.SetPartialWords(True)

        # Accumulate finalized segment texts + final result
        segments = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                try:
                    seg = json.loads(rec.Result()).get("text", "")
                except json.JSONDecodeError:
                    seg = ""
                if seg:
                    segments.append(seg)

        # Append the final result
        try:
            final_text = json.loads(rec.FinalResult()).get("text", "")
        except json.JSONDecodeError:
            final_text = ""

    # Join all parts into a single string
    all_text = " ".join([*segments, final_text]).strip()
    return all_text

# Optional: simple CLI usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path/to/audio.wav>")
        sys.exit(1)
    try:
        #start_time = time.perf_counter()
        text = speech_to_text(sys.argv[1])
        #end_time = time.perf_counter()
        print(text)
        #print(f"Transcription took {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
