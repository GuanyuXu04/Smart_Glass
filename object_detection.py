# Real-time obstacle detection with YOLO, send haptic feedback to ESP32.
import socket
import struct
import time
from collections import deque
import os
import cv2
import numpy as np
from ultralytics import YOLO

HOST = "192.168.4.1"
VIDEO_PORT = 2000
VIBRO_PORT = 4000
MODEL_PATH = "YOLO/yolo11n.engine"
CONF_THRES = 0.25
IOU_THRES = 0.45

LEFT_EDGE = 1/3
RIGHT_EDGE = 2/3
MIDDLE_SCALE = 0.75
GAMMA = 0.8
OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "stroller", "wheelchair", "traffic light", "stop sign", "bench",
    "skateboard", "scooter", "fire hydrant"
}

os.makedirs("temp", exist_ok=True)

def recvall(sock, n):
    # Read exactly n bytes from socket
    buf = b""
    while len(buf) < n:
        pkt = sock.recv(n - len(buf))
        if not pkt:
            return None
        buf += pkt
    return buf

def read_mjpeg_frame(sock):
    # Read MJPEG: 4-byte big-endian length + JPEG payload
    hdr = recvall(sock, 4)
    if hdr is None:
        return None
    (length,) = struct.unpack(">I", hdr)
    data = recvall(sock, length)
    if data is None:
        return None
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

def send_vibro_speed(sock, left, right):
    # Send LLLRRR format: 3-digit left speed + 3-digit right speed
    left = max(0, min(int(left), 999))
    right = max(0, min(int(right), 999))
    msg = f"{left:03d}{right:03d}"
    sock.sendall(msg.encode("ascii"))

def area_to_speed(area_frac, gamma=GAMMA):
    # Convert bbox area to vibration speed (0-100)
    area_frac = float(np.clip(area_frac, 0.0, 1.0))
    return int(round((area_frac ** gamma) * 100))

def vibro_from_detections(result, frame_shape):
    # Compute left/right vibro speeds based on detected obstacles
    H, W = frame_shape[:2]
    frame_area = float(H * W)

    left_speed = 0
    right_speed = 0

    if result.boxes is None or len(result.boxes) == 0:
        return left_speed, right_speed, []

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.int().cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    names = result.names  # dict: {id: "name"}

    kept = []  # for debug/overlay info

    for (x1, y1, x2, y2), cls_id, conf in zip(boxes_xyxy, classes, confs):
        name = names.get(int(cls_id), str(cls_id)).lower()
        if name not in OBSTACLE_CLASSES:
            continue

        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area_frac = (w * h) / frame_area
        speed = area_to_speed(area_frac)

        cx = (x1 + x2) / 2.0
        cx_frac = cx / W

        if cx_frac < LEFT_EDGE:
            left_speed = max(left_speed, speed)
            kept.append((name, conf, "left", speed))
        elif cx_frac > RIGHT_EDGE:
            right_speed = max(right_speed, speed)
            kept.append((name, conf, "right", speed))
        else:
            # Middle: send to both with reduced strength
            scaled = int(round(speed * MIDDLE_SCALE))
            left_speed = max(left_speed, scaled)
            right_speed = max(right_speed, scaled)
            kept.append((name, conf, "middle→both", scaled))

    return left_speed, right_speed, kept


# -------------------- Main ----------------------
def main():
    # Load YOLO model (TensorRT engine or .pt). Ultralytics accepts NumPy BGR frames.
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded.")
    
    video_socket = socket.create_connection((HOST, VIDEO_PORT))
    vibro_socket = socket.create_connection((HOST, VIBRO_PORT))
    print("✅ Connected to ESP at", f"{HOST}:{VIDEO_PORT} (video), {HOST}:{VIBRO_PORT} (vibro)")
    
    while True:
        img = read_mjpeg_frame(video_socket)
        if img is None:
            print("No frame received. Exiting.")
            break

        # Inference (returns list of Results; we expect 1 per frame)
        results = model(img, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
        result = results[0]

        # Vibromotor logic
        left_speed, right_speed, kept = vibro_from_detections(result, img.shape)

        # Print vibro commands (you'll map these to your motor driver)
        if left_speed or right_speed:
            print(f"[VIBRO] LEFT: {left_speed:3d}%, RIGHT: {right_speed:3d}% | "
                  f"objs: {', '.join(f'{n}@{s}%({side})' for n,_,side,s in kept)}")
            send_vibro_speed(vibro_socket, left_speed, right_speed)
        else:
            print("[VIBRO] No obstacles.")
            send_vibro_speed(vibro_socket, 0, 0)

        # Draw detections
        annotated = result.plot()  # returns BGR image with boxes/labels
            
        cv2.imwrite("temp/test_frame.jpg", annotated)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
