"""Train a YOLOv8 ingredient detector on the prepared cvfoodid dataset.

Designed to run on Colab/Kaggle GPUs. Example::

    python scripts/train_yolo.py \
        --data configs/data.yaml \
        --model yolov8n.pt \
        --epochs 60 \
        --imgsz 640 \
        --batch 16 \
        --project /content/drive/MyDrive/cvfoodid-runs \
        --cache disk

To resume after a Colab disconnect, re-run the same command with ``--resume``;
the script auto-detects ``{project}/{name}/weights/last.pt`` and continues
from that checkpoint. If no checkpoint exists yet, ``--resume`` is silently
ignored (so the cell is safe to leave in the notebook).

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
    parser.add_argument("--project", default=None,
                        help="Where Ultralytics writes runs/. Point this at a "
                             "Google Drive folder (e.g. "
                             "/content/drive/MyDrive/cvfoodid-runs) to survive "
                             "a Colab disconnect.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache", default=None,
                        help="Ultralytics cache mode: None, 'ram', or 'disk'. "
                             "'disk' makes epoch 2+ ~2x faster on Colab T4 "
                             "with ~1 GB extra disk.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early-stop epochs (Ultralytics default 100). "
                             "Lower this on Colab to stop before runtime cap.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from {project}/{name}/weights/last.pt if "
                             "it exists; otherwise start fresh.")
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
    if args.cache is not None:
        cfg["cache"] = args.cache
    if args.patience is not None:
        cfg["patience"] = args.patience

    # Resolve --resume to an explicit checkpoint path so the call is
    # idempotent: re-running the same cell after a disconnect picks up where
    # we left off, and re-running on a clean filesystem starts fresh.
    project_dir = Path(args.project) if args.project else Path("runs/detect")
    last_ckpt = project_dir / args.name / "weights" / "last.pt"
    if args.resume:
        if last_ckpt.is_file():
            print(f"[resume] continuing from {last_ckpt}")
            model_path: str | Path = last_ckpt
            resume_flag = True
        else:
            print(f"[resume] no checkpoint at {last_ckpt}; starting fresh.")
            model_path = args.model
            resume_flag = False
    else:
        model_path = args.model
        resume_flag = False

    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError:
        print("Ultralytics required: pip install ultralytics", file=sys.stderr)
        return 1

    model = YOLO(str(model_path))
    train_kwargs = dict(
        data=args.data,
        name=args.name,
        device=args.device,
        resume=resume_flag,
        **cfg,
    )
    if args.project is not None:
        train_kwargs["project"] = args.project
    model.train(**train_kwargs)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
