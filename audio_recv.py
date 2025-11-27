import socket
import struct
import wave
import time
from datetime import datetime

HOST = "192.168.4.1"    # ESP32 SoftAP IP (default gateway)
PORT = 1000             # Must match RECORDING_PORT in wakenet.c

# How long to wait before retrying a failed connection (seconds)
RECONNECT_DELAY = 1.0


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    """Receive exactly nbytes from socket or raise RuntimeError."""
    chunks = []
    remaining = nbytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            # Connection closed before we got everything
            raise RuntimeError(f"Socket closed with {remaining} bytes left to read")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def save_wav(data: bytes, sample_rate: int = 16000, channels: int = 1, sampwidth: int = 2) -> str:
    """Save raw PCM data to a timestamped WAV file. Returns filename."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"wake_{ts}.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(data)
    return filename

def main():
    print(f"Continuous listener started. Will connect to {HOST}:{PORT} in a loop.")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            # Try to connect to ESP
            print(f"[{datetime.now()}] Trying to connect to {HOST}:{PORT} ...", end="", flush=True)
            try:
                sock = socket.create_connection((HOST, PORT), timeout=10)
            except OSError as e:
                print(f" failed ({e}). Retrying in {RECONNECT_DELAY}s.")
                time.sleep(RECONNECT_DELAY)
                continue

            print(" connected!")
            sock.settimeout(10.0)

            try:
                # First 4 bytes: length of the recording buffer (big-endian uint32)
                header = recv_exact(sock, 4)
                (length,) = struct.unpack("!I", header)
                print(f"  Expecting {length} bytes of audio data.")

                # Read the audio data
                audio_data = recv_exact(sock, length)
                print(f"  Received {len(audio_data)} bytes.")

                # Save as WAV
                fname = save_wav(audio_data, sample_rate=16000, channels=1, sampwidth=2)
                print(f"  Saved recording to {fname}\n")

            except (RuntimeError, OSError, socket.timeout) as e:
                print(f"  Error during receive: {e}")

            finally:
                try:
                    sock.close()
                except Exception:
                    pass

            # After each full session, loop back and try to connect again.
            # ESP will open the server again after the next wakeword+10s recording.

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt: stopping listener.")
            break
        except Exception as e:
            # Catch-all to avoid the script dying; wait and retry
            print(f"Unexpected error: {e}. Retrying in {RECONNECT_DELAY}s.")
            time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    main()