#!/usr/bin/env python3
"""
flite_tts.py
------------
A simple Python wrapper for the Flite (Festival Lite) text-to-speech engine.

Usage examples:
    # Speak directly from text
    python3 flite_tts.py --text "Hello from Jetson Nano"

    # Save to a file
    python3 flite_tts.py --text "Saving this to a file" --output output.wav

    # Use a different voice
    python3 flite_tts.py --text "I am the SLT voice" --voice slt

    # Use text from a file
    python3 flite_tts.py --input textfile.txt --voice awb

Arguments:
    --text      The text to convert to speech (mutually exclusive with --input)
    --input     Path to a text file whose contents will be converted
    --output    Optional output WAV file path. If omitted, audio is played directly.
    --voice     Optional Flite voice (e.g., kal, awb, rms, slt). Default = kal
"""

import os
### Surpress Warnings from ONNX Runtime, maynot be necessary ###
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
################################################################
import argparse
import subprocess
import tempfile
import shutil
import sys


def run_command(cmd):
    """Execute a shell command and handle errors."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {' '.join(cmd)}")
        print(e)
        sys.exit(1)


def flite_tts(text, output_path=None, voice="kal"):
    """Generate or play speech using Flite."""
    # Ensure flite is installed
    if shutil.which("flite") is None:
        print("[ERROR] 'flite' command not found. Install it using:")
        print("  sudo apt install flite")
        sys.exit(1)

    # Sanitize text
    safe_text = text.replace('"', ' ')

    if output_path:
        # Save speech to WAV file
        cmd = ["flite", "-voice", voice, "-t", safe_text, "-o", output_path]
        print(f"[INFO] Generating speech to {output_path} with voice '{voice}'...")
        run_command(cmd)
        print(f"[INFO] Done! Saved to {output_path}")
    else:
        # Generate speech to temp file and play it
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmp_path = tmpfile.name
        cmd = ["flite", "-voice", voice, "-t", safe_text, "-o", tmp_path]
        print(f"[INFO] Generating speech using voice '{voice}'...")
        run_command(cmd)
        print(f"[INFO] Playing audio...")
        run_command(["aplay", tmp_path])
        os.remove(tmp_path)

def _detect_language(text: str) -> str:
    """
    Very simple heuristic language detector:
    - Returns "zh" if any CJK Unified Ideograph is present (Chinese).
    - Returns "en" if there are alphabetic characters and all are ASCII.
    - Returns "en" if there are no letters (only digits/punctuation/whitespace).
    - Returns "other" otherwise.
    """
    if not text:
        return "en"

    has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in text)
    if has_cjk:
        return "zh"

    has_alpha = any(ch.isalpha() for ch in text)
    all_ascii = all(ord(ch) < 128 for ch in text)

    if has_alpha and all_ascii:
        return "en"

    if not has_alpha:
        # Only numbers / punctuation — treat as English for TTS purposes
        return "en"

    return "other"

def piper_tts(text, output_path):
    # Ensure piper is installed
    try: 
        import wave
        from piper import PiperVoice
    except ImportError:
        print("[ERROR] 'piper' module not found. Install it using:")
        print("  pip install piper-tts")
        sys.exit(1)

    # Detect language and choose voice accordingly
    lang = _detect_language(text)
    if lang == "en":
        voice = "en_US-lessac-high"
    elif lang == "zh":
        voice = "zh_CN-huayan-medium"
    else:
        print(f"[WARNING] Could not confidently detect language for text: {text}")
        print("Defaulting to English voice.")
        voice = "en_US-lessac-high"
    print(f"[INFO] Detected language: {lang}")

    # Load the specified voice
    voice_path = f"/home/team15/piper_tts/{voice}.onnx"
    try:
        voice = PiperVoice.load(voice_path, use_cuda=False)
    except Exception as e:
        print(f"[ERROR] Could not load voice from {voice_path}: {e}")
        sys.exit(1)
    
    # Synthesize speech
    safe_text = text.replace('"', ' ')
    with wave.open(output_path, "wb") as wav_file:
        voice.synthesize_wav(safe_text, wav_file)
    
    print(f"[INFO] Generated speech saved to {output_path}")



def main():
    parser = argparse.ArgumentParser(description="Flite TTS wrapper for Jetson Nano / Linux")

    # Mutually exclusive input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text to convert to speech")
    group.add_argument("--input", type=str, help="Path to text file to read from")

    parser.add_argument("--output", type=str, help="Output WAV file path (optional)")
    parser.add_argument("--voice", type=str, default="kal", help="Voice name (e.g., kal, slt, awb, rms)")

    args = parser.parse_args()

    # Get the text either from --text or from file
    if args.input:
        if not os.path.isfile(args.input):
            print(f"[ERROR] Input file '{args.input}' does not exist.")
            sys.exit(1)
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if not text:
                print(f"[ERROR] Input file '{args.input}' is empty.")
                sys.exit(1)
    else:
        text = args.text

    flite_tts(text, args.output, args.voice)


if __name__ == "__main__":
    main()
