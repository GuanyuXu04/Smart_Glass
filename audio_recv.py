# Continuously receive audio from ESP32 and save as WAV files.
import socket
import struct
import wave
import time
from datetime import datetime

HOST = "192.168.4.1"
PORT = 1000
RECONNECT_DELAY = 1.0


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    # Read exactly nbytes from socket, raise error if connection closes early
    chunks = []
    remaining = nbytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError(f"Socket closed with {remaining} bytes left")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def save_wav(data: bytes, sample_rate: int = 16000, channels: int = 1, sampwidth: int = 2) -> str:
    # Write PCM data to timestamped WAV file
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"wake_{ts}.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(data)
    return filename

def main():
    print(f"Listening on {HOST}:{PORT}. Press Ctrl+C to stop.\n")

    while True:
        try:
            print(f"[{datetime.now()}] Connecting to {HOST}:{PORT}...", end="", flush=True)
            try:
                sock = socket.create_connection((HOST, PORT), timeout=10)
            except OSError as e:
                print(f" failed ({e}). Retrying...")
                time.sleep(RECONNECT_DELAY)
                continue

            print(" ok")
            sock.settimeout(10.0)

            try:
                # First 4 bytes: audio buffer size (big-endian uint32)
                header = recv_exact(sock, 4)
                (length,) = struct.unpack("!I", header)
                print(f"  Expected {length} bytes")

                # Read audio data
                audio_data = recv_exact(sock, length)
                print(f"  Received {len(audio_data)} bytes")

                fname = save_wav(audio_data)
                print(f"  Saved to {fname}\n")

            except (RuntimeError, OSError, socket.timeout) as e:
                print(f"  Error: {e}")

            finally:
                try:
                    sock.close()
                except Exception:
                    pass

        except KeyboardInterrupt:
            print("\nStopping.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying...")
            time.sleep(RECONNECT_DELAY)

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt: stopping listener.")
            break
        except Exception as e:
            # Catch-all to avoid the script dying; wait and retry
            print(f"Unexpected error: {e}. Retrying in {RECONNECT_DELAY}s.")
            time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    main()