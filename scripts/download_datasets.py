"""Download and stage the public datasets used to train cvfoodid.

Run from the repo root::

    python scripts/download_datasets.py --all
    python scripts/download_datasets.py foodseg103 indonesian-bakso

Each dataset lands under ``data/raw/<name>/`` and is otherwise untouched.
Use scripts/prepare_yolo_dataset.py afterwards to convert to YOLO format.

Notes on licensing
------------------
* FoodSeg103: Apache 2.0 -- usable in commercial products.
* Padang Cuisine (Kaggle): CC0 / public domain (verify on the dataset page).
* Indonesian Food (Mendeley): CC BY 4.0.
* BEEI 2024: CC BY-SA -- commercial OK with attribution + share-alike.
* Nutrition5k: research-only -- *do not* ship in a commercial product.
* Recipe1M+: research-only.

Always re-check the source page before redistributing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    license: str
    notes: str
    handler: str  # "http_zip" | "git" | "kaggle" | "manual"


SOURCES: list[Source] = [
    Source(
        name="foodseg103",
        url="https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip",
        license="Apache-2.0",
        notes="Password 'LARCdataset9947' required. Ingredient-level masks (103 classes).",
        handler="http_zip",
    ),
    Source(
        name="foodinsseg",
        url="https://github.com/jamesjg/FoodInsSeg",
        license="Academic-Only",
        notes="Instance segmentation extension of FoodSeg103.",
        handler="git",
    ),
    Source(
        name="indonesian-food-mendeley",
        url="https://data.mendeley.com/public-files/datasets/vtjd68bmwt/files/",
        license="CC-BY-4.0",
        notes="10 Indonesian dishes (bakso, gado-gado, rendang, ...).",
        handler="manual",
    ),
    Source(
        name="padang-cuisine",
        url="faldoae/padangfood",
        license="CC0",
        notes="Kaggle slug. Requires kaggle CLI authenticated.",
        handler="kaggle",
    ),
    Source(
        name="indonesian-traditional-foods",
        url="rizkyyk/dataset-food-classification",
        license="CC-BY",
        notes="Kaggle. ~ 24 Indonesian classes.",
        handler="kaggle",
    ),
    Source(
        name="eriko-syah-nutrition",
        url="https://huggingface.co/datasets/eriko-syah/indonesian-food",
        license="Unknown",
        notes="Indonesian dish-level kcal/protein/fat/carb table.",
        handler="manual",
    ),
    Source(
        name="nutrition5k",
        url="gs://nutrition5k_dataset/nutrition5k_dataset/",
        license="Research-Only",
        notes="181 GB. Use gsutil. Do not ship commercially.",
        handler="manual",
    ),
]


def _http_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "cvfoodid/0.1"})
    with urlopen(req) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def fetch_http_zip(src: Source, root: Path) -> Path:
    target = root / src.name
    target.mkdir(parents=True, exist_ok=True)
    archive = target / Path(src.url).name
    if not archive.is_file():
        print(f"[{src.name}] downloading -> {archive}")
        _http_download(src.url, archive)
    if archive.suffix == ".zip":
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
        except RuntimeError as exc:
            print(f"[{src.name}] zip extract failed (likely password-protected): {exc}")
            print(f"  Manual: unzip -P <password> {archive} -d {target}")
    return target


def fetch_git(src: Source, root: Path) -> Path:
    target = root / src.name
    if (target / ".git").is_dir():
        print(f"[{src.name}] already cloned at {target}")
        return target
    cmd = ["git", "clone", "--depth", "1", src.url, str(target)]
    print(f"[{src.name}] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return target


def fetch_kaggle(src: Source, root: Path) -> Path:
    target = root / src.name
    target.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", src.url, "-p", str(target), "--unzip"]
    print(f"[{src.name}] running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("kaggle CLI not found. Install with: pip install kaggle")
        print("Then put your token at ~/.kaggle/kaggle.json")
        raise
    return target


def fetch_manual(src: Source, root: Path) -> Path:
    target = root / src.name
    target.mkdir(parents=True, exist_ok=True)
    readme = target / "DOWNLOAD.md"
    readme.write_text(
        f"# {src.name}\n\n"
        f"Source: {src.url}\n\nLicense: {src.license}\n\nNotes: {src.notes}\n\n"
        "Download manually from the URL above and place files in this directory.\n",
        encoding="utf-8",
    )
    print(f"[{src.name}] manual download required. See {readme}")
    return target


def fetch(src: Source, root: Path) -> Path:
    if src.handler == "http_zip":
        return fetch_http_zip(src, root)
    if src.handler == "git":
        return fetch_git(src, root)
    if src.handler == "kaggle":
        return fetch_kaggle(src, root)
    return fetch_manual(src, root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Dataset names to fetch.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit.")
    parser.add_argument("--root", default="data/raw", help="Output directory.")
    args = parser.parse_args()

    if args.list:
        print(json.dumps([s.__dict__ for s in SOURCES], indent=2))
        return 0

    selected = SOURCES if args.all else [s for s in SOURCES if s.name in args.names]
    if not selected:
        parser.error("No dataset selected. Use --all, --list, or pass names.")
    root = Path(args.root)
    for src in selected:
        try:
            fetch(src, root)
        except Exception as exc:  # pragma: no cover - network paths
            print(f"[{src.name}] FAILED: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
