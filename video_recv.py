# Receive video frames from ESP32 over MJPEG stream and save to disk.
import socket, struct, cv2, numpy as np

HOST = "192.168.4.1"
PORT = 2000

def recvall(sock, n):
    # Read exactly n bytes from socket, return None if connection drops
    buf = b''
    while len(buf) < n:
        pkt = sock.recv(n - len(buf))
        if not pkt:
            return None
        buf += pkt
    return buf


with socket.create_connection((HOST, PORT)) as s:
    print("Connected to ESP32")
    while True:
        hdr = recvall(s, 4)
        if hdr is None:
            break
        (length,) = struct.unpack('>I', hdr)
        data = recvall(s, length)
        if data is None:
            break
        # Decode JPEG and write to temp file
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite("temp/test_frame.jpg", img)
        if img is None:
            continue
cv2.destroyAllWindows()
