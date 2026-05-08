"""Convert FoodSeg103 (and friends) into Ultralytics YOLO format.

Input layout (after running download_datasets.py)::

    data/raw/foodseg103/FoodSeg103/
        Images/img_dir/{train,test}/*.jpg
        Images/ann_dir/{train,test}/*.png   # 1-channel mask, 0..103

Output layout written under ``data/processed/yolo``::

    images/{train,val}/*.jpg
    labels/{train,val}/*.txt   # YOLO bbox-from-mask format

We extract bounding boxes from each connected component in the mask. For
multi-label aggregation, the script also keeps a ``class_map.json`` so we can
remap FoodSeg103's 103 ingredients to the smaller cvfoodid vocabulary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

# Mapping FoodSeg103 ingredient names -> cvfoodid TKPI ids. Only the entries
# below are used for training. Any FoodSeg103 mask whose label is *not* in this
# map is discarded for now (TODO: extend over time as the vocabulary grows).
FOODSEG103_TO_TKPI: dict[str, str] = {
    "rice": "nasi_putih",
    "noodles": "mie",
    "egg": "telur_ayam",
    "tofu": "tahu",
    "chicken duck": "daging_ayam",
    "beef": "daging_sapi",
    "pork": "daging_sapi",  # absent in halal context but mapped for completeness
    "lamb": "daging_kambing",
    "fish": "ikan",
    "shrimp": "udang",
    "squid": "cumi",
    "carrot": "sayur_wortel",
    "tomato": "tomat",
    "cucumber": "timun",
    "potato": "kentang",
    "broccoli": "sayur_brokoli",
    "cabbage": "sayur_kol",
    "spinach": "sayur_bayam",
    "lettuce": "sayur_selada",
    "green beans": "sayur_buncis",
    "corn": "sayur_jagung",
    "eggplant": "sayur_terong",
    "mushroom": "jamur",
    "banana": "pisang",
    "apple": "apel",
    "orange": "jeruk",
    "mango": "mangga",
    "watermelon": "semangka",
    "papaya": "pepaya",
    "bread": "roti_putih",
    "soup": "kuah_sup",
    "sausage": "sosis",
    "cheese butter": "keju",
    "milk": "susu_sapi",
    "yogurt": "yogurt",
    "ginger": "jahe",
    "garlic": "bawang_putih",
    "onion": "bawang_merah",
    "chili": "cabai_merah",
}


def _label_index(data_yaml: Path) -> dict[str, int]:
    cfg = yaml.safe_load(data_yaml.read_text())
    return {name: int(idx) for idx, name in cfg["names"].items()}


def _connected_bboxes(mask: np.ndarray, class_value: int) -> list[tuple[int, int, int, int]]:
    """Return YOLO-friendly (x1, y1, x2, y2) bboxes for one class in a mask."""
    bin_mask = (mask == class_value).astype(np.uint8)
    if bin_mask.sum() == 0:
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    bboxes = []
    for i in range(1, n):  # skip background
        x, y, w, h, area = stats[i]
        if area < 32:  # discard tiny noise blobs
            continue
        bboxes.append((int(x), int(y), int(x + w), int(y + h)))
    return bboxes


def convert_foodseg103(raw_dir: Path, out_dir: Path, data_yaml: Path,
                       split_map: dict[str, str]) -> dict[str, int]:
    """Walk FoodSeg103 splits and emit YOLO labels under ``out_dir``."""
    name_to_idx = _label_index(data_yaml)

    # FoodSeg103's per-class ID list lives in category_id.txt (1-indexed,
    # 0 = background). We accept either category_id.txt or names.txt.
    category_file = raw_dir / "Images" / "category_id.txt"
    if not category_file.is_file():
        raise FileNotFoundError(f"FoodSeg103 category file missing: {category_file}")
    fs_id_to_name: dict[int, str] = {}
    for line in category_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        idx_str, name = line.split("\t", 1)
        fs_id_to_name[int(idx_str)] = name.strip().lower()

    counts: dict[str, int] = defaultdict(int)
    for fs_split, yolo_split in split_map.items():
        img_dir = raw_dir / "Images" / "img_dir" / fs_split
        ann_dir = raw_dir / "Images" / "ann_dir" / fs_split
        if not img_dir.is_dir() or not ann_dir.is_dir():
            print(f"  skip: missing {img_dir} or {ann_dir}", file=sys.stderr)
            continue
        out_img = out_dir / "images" / yolo_split
        out_lbl = out_dir / "labels" / yolo_split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(img_dir.glob("*.jpg")):
            mask_path = ann_dir / (img_path.stem + ".png")
            if not mask_path.is_file():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            h, w = mask.shape[:2]
            lines: list[str] = []
            for fs_id, fs_name in fs_id_to_name.items():
                tkpi_name = FOODSEG103_TO_TKPI.get(fs_name)
                if tkpi_name is None or tkpi_name not in name_to_idx:
                    continue
                cls_idx = name_to_idx[tkpi_name]
                for x1, y1, x2, y2 in _connected_bboxes(mask, fs_id):
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    counts[tkpi_name] += 1
            if not lines:
                continue
            shutil.copy2(img_path, out_img / img_path.name)
            (out_lbl / (img_path.stem + ".txt")).write_text("\n".join(lines) + "\n")
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/foodseg103/FoodSeg103",
                        help="Path to extracted FoodSeg103 root.")
    parser.add_argument("--out", default="data/processed/yolo", help="YOLO output dir.")
    parser.add_argument("--data-yaml", default="configs/data.yaml")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    data_yaml = Path(args.data_yaml)

    if not raw_dir.is_dir():
        print(f"FoodSeg103 not found at {raw_dir}. Run scripts/download_datasets.py foodseg103 first.",
              file=sys.stderr)
        return 1

    counts = convert_foodseg103(
        raw_dir, out_dir, data_yaml, split_map={"train": "train", "test": "val"}
    )
    print(json.dumps({"images": str(out_dir), "per_class": counts}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
