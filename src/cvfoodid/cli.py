"""Command-line entry points exposed in ``pyproject.toml``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cvfoodid.i18n import Lang, label
from cvfoodid.nutrition.database import NutritionDatabase


def _print_summary(result_dict: dict[str, object], lang: Lang) -> None:
    total = result_dict["total"]  # type: ignore[index]
    ings = result_dict["ingredients"]  # type: ignore[index]
    print(f"--- {label('total', lang)} ---")
    print(f"  {label('mass_g', lang)}: {total['mass_g']}")  # type: ignore[index]
    print(f"  {label('kcal', lang)}: {total['kcal']}")      # type: ignore[index]
    print(f"  {label('protein', lang)}: {total['protein_g']}")  # type: ignore[index]
    print(f"  {label('fat', lang)}: {total['fat_g']}")          # type: ignore[index]
    print(f"  {label('carb', lang)}: {total['carb_g']}")        # type: ignore[index]
    print(f"--- {label('ingredient', lang)} ---")
    for line in ings:  # type: ignore[union-attr]
        name = line["name_id"] if lang == Lang.ID else line["name_en"]  # type: ignore[index]
        print(f"  {name}: {line['mass_g']} g, {line['kcal']} kcal")  # type: ignore[index]


def detect_cmd(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cvfoodid-detect",
        description="Run end-to-end detection + nutrition estimation on an image.",
    )
    parser.add_argument("--image", required=True, help="Path to the food image.")
    parser.add_argument("--weights", required=True, help="Path to YOLO weights (.pt).")
    parser.add_argument("--plate-mm", type=float, default=240.0,
                        help="Fallback plate diameter in mm (default 240).")
    parser.add_argument("--lang", choices=["id", "en"], default="id")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args(argv)

    # Lazy-import the detector and pipeline so the help text works without
    # ultralytics installed.
    from cvfoodid.detection.detector import IngredientDetector
    from cvfoodid.pipeline import FoodPipeline

    detector = IngredientDetector(weights_path=args.weights, conf=args.conf)
    pipeline = FoodPipeline(detector=detector, plate_diameter_mm=args.plate_mm)
    out = pipeline.run(args.image)
    payload = out.as_dict()
    if args.json:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0
    _print_summary(payload, Lang(args.lang))
    return 0


def train_cmd(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cvfoodid-train",
        description="Train a YOLO ingredient detector. Thin wrapper over Ultralytics.",
    )
    parser.add_argument("--data", required=True, help="Path to data.yaml.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model or .yaml.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="cvfoodid-yolo")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError:
        print("Ultralytics is required: pip install ultralytics", file=sys.stderr)
        return 1
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
    )
    return 0


def db_info_cmd(argv: list[str] | None = None) -> int:
    """Print a quick summary of the bundled nutrition database."""
    parser = argparse.ArgumentParser(prog="cvfoodid-db", description=db_info_cmd.__doc__)
    parser.add_argument("--csv", default=None, help="Override CSV path.")
    args = parser.parse_args(argv)
    db = NutritionDatabase(Path(args.csv) if args.csv else None)
    print(f"Loaded {len(db)} ingredients from {db.csv_path}")
    for cat in sorted({db.require(i).category for i in db.all_ids()}):
        print(f"  {cat}: {len(db.by_category(cat))}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(detect_cmd())
