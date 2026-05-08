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

Usage::

    python scripts/foodseg103_from_hf.py
    python scripts/foodseg103_from_hf.py --limit 50  # smoke-test
"""

from __future__ import annotations

import argparse
import json
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


def _download_parquets(repo: str, files: Iterable[str]) -> list[Path]:
    from huggingface_hub import hf_hub_download

    paths = []
    for fname in files:
        print(f"  fetching {fname} ...", flush=True)
        local = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
        paths.append(Path(local))
    return paths


def _emit_split(parquet_paths: list[Path], split: str, out_root: Path,
                limit: int | None) -> int:
    img_dir = out_root / "Images" / "img_dir" / split
    ann_dir = out_root / "Images" / "ann_dir" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in parquet_paths:
        table = pq.read_table(path, columns=["image", "label", "id"])
        n = table.num_rows
        for i in range(n):
            if limit is not None and written >= limit:
                return written
            row = table.slice(i, 1).to_pylist()[0]
            image_struct = row["image"]
            label_struct = row["label"]
            row_id = row["id"]

            stem = (image_struct.get("path") or f"{row_id:08d}.jpg").rsplit(".", 1)[0]
            (img_dir / f"{stem}.jpg").write_bytes(image_struct["bytes"])
            (ann_dir / f"{stem}.png").write_bytes(label_struct["bytes"])
            written += 1
            if written % 500 == 0:
                print(f"    wrote {written} {split} pairs ...", flush=True)
    return written


def _write_category_file(out_root: Path) -> None:
    """Reconstruct Images/category_id.txt from HF id2label.json."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id=HF_REPO, filename="id2label.json",
                            repo_type="dataset")
    id2label = json.loads(Path(local).read_text(encoding="utf-8"))

    out_path = out_root / "Images" / "category_id.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for idx_str, name in sorted(id2label.items(), key=lambda kv: int(kv[0])):
        idx = int(idx_str)
        if idx == 0:  # background sentinel - skip in category file
            continue
        lines.append(f"{idx}\t{name}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out_path} ({len(lines)} classes)")


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

    _write_category_file(out_root)
    for split in args.splits:
        print(f"split={split}")
        parquets = _download_parquets(HF_REPO, HF_PARQUETS[split])
        n = _emit_split(parquets, split, out_root, args.limit)
        print(f"  {split}: {n} (image, mask) pairs written")
    print(f"done. layout at {out_root}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
