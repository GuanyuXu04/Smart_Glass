from pathlib import Path
import argparse

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    cv2 = None  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    YOLO = None


def run_detection_demo(image_path: str, model_path: str):
    """Run a single-image YOLO inference and print box info.

    Paths should be absolute or relative to the current working directory.
    The script's CLI supplies sensible defaults resolved relative to the
    repository root so running `python tests/test_YOLO.py` from repo root
    will work.
    """
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is not installed.")

    if cv2 is None:
        raise RuntimeError("OpenCV is not installed.")

    model = YOLO(model_path)
    results = model(image_path)

    for result in results:
        xywh = result.boxes.xywh
        print("center coordinates (x, y), width and height:\n", xywh)
        xywhn = result.boxes.xywhn
        print("normalized center coordinates (x, y), width and height:\n", xywhn)
        xyxy = result.boxes.xyxy
        print("top-left and bottom-right coordinates (x, y):\n", xyxy)
        xyxyn = result.boxes.xyxyn
        print("normalized top-left and bottom-right coordinates (x, y):\n", xyxyn)
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        print("class names: ", names)
        confs = result.boxes.conf
        print("confidence scores: ", confs)

        result_plotted = result.plot()
        cv2.imwrite("output.jpg", result_plotted)

def run_segmentation_demo(image_path: str, model_path: str):
    """Run a single-image YOLO segmentation inference and print mask info.

    Paths should be absolute or relative to the current working directory.
    The script's CLI supplies sensible defaults resolved relative to the
    repository root so running `python tests/test_YOLO.py` from repo root
    will work.
    """
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is not installed.")

    if cv2 is None:
        raise RuntimeError("OpenCV is not installed.")

    model = YOLO(model_path)
    results = model(image_path)

    for result in results:
        masks = result.masks.data
        print("mask data:\n", masks)
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        print("class names: ", names)
        confs = result.boxes.conf
        print("confidence scores: ", confs)

        result_plotted = result.plot()
        cv2.imwrite("output_segmentation.jpg", result_plotted)


def _default_paths():
    # Resolve repo root relative to this test file, so running from repo root
    # or from another CWD works consistently.
    repo_root = Path(__file__).resolve().parent.parent
    default_image = str((repo_root / "images" / "room.jpg").resolve())
    # Prefer a .pt (PyTorch) or .onnx model if available. A .engine file requires
    # TensorRT which may not be present in the environment.
    model_dir = repo_root / "YOLO"
    candidates = [      
        model_dir / "yolo11n-seg.engine",
    ]
    for c in candidates:
        if c.exists():
            return default_image, str(c.resolve())
    # If none found, still return the engine path (to preserve previous behavior)
    return default_image, str((model_dir / "yolo11n-seg.engine").resolve())


if __name__ == "__main__":  # pragma: no cover - manual demo
    default_image, default_model = _default_paths()
    parser = argparse.ArgumentParser(description="Run YOLO demo on a single image")
    parser.add_argument("--image", default=default_image, help="Path to input image")
    parser.add_argument("--model", default=default_model, help="Path to YOLO model file")
    parser.add_argument("--task", default="detect", help="YOLO task: detect, segment, classify, etc.")
    args = parser.parse_args()
    run_detection_demo(args.image, args.model)
