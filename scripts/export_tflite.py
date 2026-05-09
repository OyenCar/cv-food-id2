"""Export a trained YOLO model to TensorFlow Lite for Android/iOS.

Pipeline::

    YOLO .pt -> ONNX -> TFLite (INT8 with representative dataset)

The script uses Ultralytics' built-in exporter, which handles ONNX -> TF
SavedModel -> TFLite under the hood. INT8 calibration runs on CPU
regardless of the ``--device`` flag (a TFLite stack limitation), and by
default iterates the *entire* ``val`` split. On a Kaggle/Colab CPU box
that means ~1-2 s per image -> 30-120 min for FoodSeg103's ~2 k images.

Use ``--fraction 0.05`` to sample only ~5 %% of val (~100 images) for
calibration -- usually within ~1 mAP point of full-val INT8, but ~20x
faster.

Usage::

    # FP32 -- ~1-2 min on CPU, ships fine for MVP (~12 MB .tflite).
    python scripts/export_tflite.py \
        --weights runs/detect/cvfoodid-yolo/weights/best.pt \
        --imgsz 640

    # INT8 with fast calibration (~5-10 min on Kaggle/Colab CPU).
    python scripts/export_tflite.py \
        --weights runs/detect/cvfoodid-yolo/weights/best.pt \
        --imgsz 640 \
        --calib configs/data.yaml \
        --int8 \
        --fraction 0.05
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
                             "calibration. Ultralytics samples images "
                             "from the YAML's ``val`` split. Pass the "
                             "same YAML you trained with, e.g. "
                             "configs/data.yaml.")
    parser.add_argument("--fraction", type=float, default=None,
                        help="Fraction of the val split to use for INT8 "
                             "calibration (0.0-1.0). 0.05 (~5%%) is a "
                             "reasonable default on Kaggle/Colab CPU; "
                             "omit to use 100%% (slow). Only used when "
                             "--int8 is set.")
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
        if args.fraction is not None:
            if not 0.0 < args.fraction <= 1.0:
                print(
                    f"--fraction must be in (0.0, 1.0], got {args.fraction}",
                    file=sys.stderr,
                )
                return 1
            kwargs["fraction"] = args.fraction

    out = model.export(**kwargs)
    print(f"Exported: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
