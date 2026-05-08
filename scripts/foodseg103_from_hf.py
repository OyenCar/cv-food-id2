"""Download FoodSeg103 from a HuggingFace mirror and stage it in the
SMU-original folder layout that ``scripts/prepare_yolo_dataset.py`` expects.

Why this exists
---------------
The canonical SMU URL (``research.larc.smu.edu.sg``) routinely times out from
Colab IPs. The HuggingFace mirror at ``EduardoPacheco/FoodSeg103`` ships the
same images and per-pixel masks but in parquet form. This script fetches the
parquet shards and re-emits the original layout so the rest of the pipeline
does not need to know which mirror was used.

Output layout (matches the SMU original)::

    data/raw/foodseg103/FoodSeg103/
        Images/
            category_id.txt              <-- "{idx}\\t{name}" per line, 1-indexed
            img_dir/{train,test}/*.jpg   <-- RGB images
            ann_dir/{train,test}/*.png   <-- 1-channel masks (0..103)

``Images/category_id.txt`` is written **last** and via an atomic rename, so the
notebook can use it as a reliable "fully extracted" sentinel: a partial run
leaves no marker, and a re-run will resume cleanly.

Usage::

    python scripts/foodseg103_from_hf.py
    python scripts/foodseg103_from_hf.py --limit 50  # smoke-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import pyarrow.parquet as pq

HF_REPO = "EduardoPacheco/FoodSeg103"
HF_PARQUETS = {
    "train": [
        "data/train-00000-of-00003.parquet",
        "data/train-00001-of-00003.parquet",
        "data/train-00002-of-00003.parquet",
    ],
    # SMU calls the held-out split "test"; HF calls it "validation". We
    # write under "test" so the prep script's split_map continues to work.
    "test": ["data/validation-00000-of-00001.parquet"],
}

# Read at most this many rows into memory at a time. Each row is roughly an
# image (~50-200 KB) plus a mask (~5-20 KB), so 256 rows is well under 100 MB.
ROW_BATCH = 256


def _safe_stem(raw: str | None, fallback_id: int) -> str:
    """Drop any path separators or '..' segments to prevent traversal."""
    if not raw:
        return f"{fallback_id:08d}"
    base = os.path.basename(str(raw))
    stem = base.rsplit(".", 1)[0]
    # If the caller fed in something pathological (e.g. all separators)
    # base could be empty. Fall back to the row id in that case.
    return stem or f"{fallback_id:08d}"


def _download_parquets(repo: str, files: Iterable[str]) -> list[Path]:
    from huggingface_hub import hf_hub_download

    paths = []
    for fname in files:
        print(f"  fetching {fname} ...", flush=True)
        local = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
        paths.append(Path(local))
    return paths


def _emit_split(
    parquet_paths: list[Path], split: str, out_root: Path, limit: int | None
) -> int:
    img_dir = out_root / "Images" / "img_dir" / split
    ann_dir = out_root / "Images" / "ann_dir" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=ROW_BATCH,
                                     columns=["image", "label", "id"]):
            for row in batch.to_pylist():
                if limit is not None and written >= limit:
                    return written

                image_struct = row["image"]
                label_struct = row["label"]
                row_id = int(row["id"])

                stem = _safe_stem(image_struct.get("path"), row_id)
                img_path = img_dir / f"{stem}.jpg"
                ann_path = ann_dir / f"{stem}.png"

                # Resume-friendly: skip pairs we've already emitted.
                if not img_path.exists():
                    img_path.write_bytes(image_struct["bytes"])
                if not ann_path.exists():
                    ann_path.write_bytes(label_struct["bytes"])

                written += 1
                if written % 500 == 0:
                    print(f"    wrote {written} {split} pairs ...", flush=True)
    return written


def _write_category_file_atomic(out_root: Path) -> None:
    """Reconstruct ``Images/category_id.txt`` from HF ``id2label.json``.

    Writes ``category_id.txt.tmp`` first then atomically renames it. A crashed
    run leaves no ``category_id.txt`` so the notebook's sentinel check
    correctly triggers a re-extract on the next run.
    """
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id=HF_REPO, filename="id2label.json",
                            repo_type="dataset")
    id2label = json.loads(Path(local).read_text(encoding="utf-8"))

    out_path = out_root / "Images" / "category_id.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for idx_str, name in sorted(id2label.items(), key=lambda kv: int(kv[0])):
        idx = int(idx_str)
        if idx == 0:  # background sentinel; FoodSeg103 categories are 1..103
            continue
        lines.append(f"{idx}\t{name}")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)
    print(f"  wrote {out_path} ({len(lines)} classes)")


def _summarise(out_root: Path) -> None:
    img_root = out_root / "Images" / "img_dir"
    ann_root = out_root / "Images" / "ann_dir"
    for split in ("train", "test"):
        img_split = img_root / split
        ann_split = ann_root / split
        n_img = len(list(img_split.glob("*.jpg"))) if img_split.is_dir() else 0
        n_ann = len(list(ann_split.glob("*.png"))) if ann_split.is_dir() else 0
        print(f"  summary: {split}: {n_img} images, {n_ann} masks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/foodseg103/FoodSeg103",
                        help="Target SMU-style root.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N rows per split (smoke testing).")
    parser.add_argument("--splits", nargs="+", default=list(HF_PARQUETS.keys()),
                        choices=list(HF_PARQUETS.keys()),
                        help="Subset of splits to fetch.")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Extract images first; only stamp category_id.txt at the very end so
    # partial failures leave no marker and the notebook re-runs cleanly.
    for split in args.splits:
        print(f"split={split}")
        parquets = _download_parquets(HF_REPO, HF_PARQUETS[split])
        n = _emit_split(parquets, split, out_root, args.limit)
        print(f"  {split}: {n} (image, mask) pairs written")

    _write_category_file_atomic(out_root)
    _summarise(out_root)
    print(f"done. layout at {out_root}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
