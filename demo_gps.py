#!/usr/bin/env python3
# GPS-based navigation demo with YOLO obstacle detection visualization.
# Adapted from: https://github.com/janChen0310/OSM-Valhalla-Routing-Demo

import socket
import struct
import time
import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple
import logging

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

try:
    import requests
except ImportError:
    requests = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HOST = "192.168.4.1"
VIDEO_PORT = 2000
YOLO_MODEL = "yolo11n.pt"
CONF_THRES = 0.25
IOU_THRES = 0.45
OUTPUT_DIR = Path(".")
OUTPUT_IMG = OUTPUT_DIR / "output.jpg"

# Valhalla API defaults
VALHALLA_HOST = "127.0.0.1"
VALHALLA_PORT = 8002


def recv_exact(sock: socket.socket, nbytes: int) -> Optional[bytes]:
    # Read exactly nbytes from socket
    buf = b""
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_mjpeg_frame(sock: socket.socket) -> Optional[np.ndarray]:
    # Read MJPEG: 4-byte big-endian length + JPEG payload
    hdr = recv_exact(sock, 4)
    if hdr is None:
        return None
    (length,) = struct.unpack(">I", hdr)
    data = recv_exact(sock, length)
    if data is None:
        return None
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def run_yolo_inference(
    model: YOLO, img: np.ndarray, conf: float = CONF_THRES, iou: float = IOU_THRES
) -> Tuple[np.ndarray, list]:
    # Run YOLO detection and return annotated image + detection list
    results = model.predict(img, conf=conf, iou=iou)
    result = results[0]
    
    annotated = result.plot()
    detections = []
    
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_val = box.conf.item()
            cls_id = int(box.cls.item())
            cls_name = result.names.get(cls_id, f"Class {cls_id}")
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(conf_val),
                "class": cls_name,
                "class_id": cls_id,
            })
    
    return annotated, detections


def query_valhalla_route(
    start: Tuple[float, float],
    end: Tuple[float, float],
    valhalla_host: str = VALHALLA_HOST,
    valhalla_port: int = VALHALLA_PORT,
) -> Optional[dict]:
    """Query Valhalla for route between two GPS coordinates (lat, lon)."""
    if requests is None:
        logger.warning("requests not installed; skipping route query")
        return None
    
    url = f"http://{valhalla_host}:{valhalla_port}/route"
    params = {
        "json": json.dumps({
            "locations": [
                {"lat": start[0], "lon": start[1]},
                {"lat": end[0], "lon": end[1]},
            ],
            "costing": "pedestrian",
            "format": "json",
        })
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Valhalla query failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="GPS navigation + YOLO detection demo")
    parser.add_argument("--host", default=HOST, help="ESP32 IP")
    parser.add_argument("--video-port", type=int, default=VIDEO_PORT, help="Video port")
    parser.add_argument("--yolo-model", default=YOLO_MODEL, help="YOLO model path")
    parser.add_argument("--conf", type=float, default=CONF_THRES, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=IOU_THRES, help="YOLO IoU threshold")
    parser.add_argument("--output", default=str(OUTPUT_IMG), help="Output image path")
    parser.add_argument("--valhalla-host", default=VALHALLA_HOST, help="Valhalla API host")
    parser.add_argument("--valhalla-port", type=int, default=VALHALLA_PORT, help="Valhalla API port")
    parser.add_argument("--start-lat", type=float, help="Start latitude")
    parser.add_argument("--start-lon", type=float, help="Start longitude")
    parser.add_argument("--end-lat", type=float, help="End latitude")
    parser.add_argument("--end-lon", type=float, help="End longitude")
    args = parser.parse_args()

    logger.info(f"Connecting to {args.host}:{args.video_port}")
    
    # Load YOLO
    model = None
    if _YOLO_AVAILABLE:
        try:
            model = YOLO(args.yolo_model)
            logger.info(f"YOLO model loaded: {args.yolo_model}")
        except Exception as e:
            logger.warning(f"Failed to load YOLO: {e}")
    else:
        logger.warning("ultralytics not available; YOLO detection disabled")

    # Query route if coordinates provided
    if args.start_lat is not None and args.end_lat is not None:
        logger.info(f"Querying route from ({args.start_lat}, {args.start_lon}) to ({args.end_lat}, {args.end_lon})")
        route = query_valhalla_route(
            (args.start_lat, args.start_lon),
            (args.end_lat, args.end_lon),
            args.valhalla_host,
            args.valhalla_port,
        )
        if route:
            logger.info("Route retrieved successfully")
            legs = route.get("trip", {}).get("legs", [])
            for i, leg in enumerate(legs):
                summary = leg.get("summary", {})
                logger.info(f"Leg {i+1}: {summary.get('distance', 0)/1000:.2f} km, {summary.get('time', 0):.0f}s")
        else:
            logger.info("No route data available")

    # Receive and process video frames
    try:
        with socket.create_connection((args.host, args.video_port), timeout=5) as sock:
            logger.info(f"Connected to {args.host}:{args.video_port}")
            frame_count = 0
            
            while True:
                img = recv_mjpeg_frame(sock)
                if img is None:
                    logger.warning("Failed to receive frame")
                    break
                
                frame_count += 1
                logger.info(f"Received frame {frame_count}: shape={img.shape}")
                
                # Run YOLO if available
                if model is not None:
                    annotated, detections = run_yolo_inference(img, args.conf, args.iou)
                    logger.info(f"Detected {len(detections)} objects")
                    for det in detections:
                        logger.info(f"  - {det['class']} ({det['confidence']:.2f})")
                    
                    # Save annotated image
                    cv2.imwrite(args.output, annotated)
                    logger.info(f"Saved annotated frame to {args.output}")
                else:
                    cv2.imwrite(args.output, img)
                    logger.info(f"Saved frame to {args.output}")
                
                # Process only one frame for demo; remove this for continuous streaming
                break
    
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        logger.error(f"Connection error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted")
    
    return 0


if __name__ == "__main__":
    exit(main())
