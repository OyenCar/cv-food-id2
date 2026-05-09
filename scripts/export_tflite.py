"""Export a trained YOLO model to TensorFlow Lite for Android/iOS.

Pipeline::

    YOLO .pt -> ONNX -> TFLite (INT8 with representative dataset)

The script uses Ultralytics' built-in exporter, which handles ONNX -> TF
SavedModel -> TFLite under the hood. INT8 quantization needs a representative
dataset for calibration; Ultralytics samples it automatically from the
``val`` split declared in the dataset YAML.

Usage::

    python scripts/export_tflite.py \
        --weights runs/detect/cvfoodid-yolo/weights/best.pt \
        --imgsz 640 \
        --calib configs/data.yaml \
        --int8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--int8", action="store_true",
                        help="INT8 post-training quantization (smaller, faster on mobile).")
    parser.add_argument("--calib", default=None,
                        help="Path to the dataset YAML used for INT8 "
                             "calibration. Ultralytics samples ~100 "
                             "images from the YAML's ``val`` split. Pass "
                             "the same YAML you trained with, e.g. "
                             "configs/data.yaml.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError:
        print("Ultralytics required: pip install ultralytics", file=sys.stderr)
        return 1

    weights = Path(args.weights)
    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        return 1

    model = YOLO(str(weights))
    kwargs: dict[str, object] = {
        "format": "tflite",
        "imgsz": args.imgsz,
        "device": args.device,
    }
    if args.int8:
        kwargs["int8"] = True
        if args.calib:
            kwargs["data"] = args.calib

    out = model.export(**kwargs)
    print(f"Exported: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
