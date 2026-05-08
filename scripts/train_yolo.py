"""Train a YOLOv8 ingredient detector on the prepared cvfoodid dataset.

Designed to run on Colab/Kaggle GPUs. Example::

    python scripts/train_yolo.py \
        --data configs/data.yaml \
        --model yolov8n.pt \
        --epochs 100 \
        --imgsz 640 \
        --batch 16

The script wraps Ultralytics so we can centralize hyperparameters in
``configs/training.yaml`` and apply our defaults regardless of how the user
invokes training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Pretrained weights (.pt) or model yaml.")
    parser.add_argument("--cfg", default="configs/training.yaml",
                        help="Training hyperparameters yaml.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--name", default="cvfoodid-yolo")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.is_file():
        print(f"Training config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(cfg_path.read_text())

    # CLI overrides win over the YAML defaults.
    for k in ("epochs", "imgsz", "batch"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError:
        print("Ultralytics required: pip install ultralytics", file=sys.stderr)
        return 1

    model = YOLO(args.model)
    model.train(
        data=args.data,
        name=args.name,
        device=args.device,
        resume=args.resume,
        **cfg,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
