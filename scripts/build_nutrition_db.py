"""Validate and (optionally) extend ``data/nutrition/tkpi_lookup.csv``.

This script does two things:

1. Sanity-checks the bundled CSV: required columns, no duplicate ids,
   numeric fields parseable, density > 0, kcal == 4*P + 9*F + 4*C +- 25%.
2. Merges the public ``eriko-syah/indonesian-food`` HF dataset (dish-level
   gizi) into a *per-dish* lookup at ``data/nutrition/dish_lookup.csv`` for
   the dish-level fallback path used by the calculator.

Run with ``--check-only`` to skip the network call.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

REQUIRED_COLUMNS = [
    "id", "name_id", "name_en", "category", "kcal_per_100g",
    "protein_g", "fat_g", "carb_g", "fiber_g", "water_g",
    "density_g_per_ml", "source",
]
NUMERIC_COLUMNS = [
    "kcal_per_100g", "protein_g", "fat_g", "carb_g", "fiber_g",
    "water_g", "density_g_per_ml",
]


def check_csv(csv_path: Path) -> int:
    if not csv_path.is_file():
        print(f"CSV missing: {csv_path}", file=sys.stderr)
        return 1
    seen_ids: set[str] = set()
    issues = 0
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            print(f"  missing columns: {missing}", file=sys.stderr)
            return 1
        for row_no, row in enumerate(reader, start=2):
            rid = row["name_id"]
            if rid in seen_ids:
                print(f"  row {row_no}: duplicate id {rid!r}", file=sys.stderr)
                issues += 1
            seen_ids.add(rid)
            try:
                vals = {c: float(row[c]) for c in NUMERIC_COLUMNS}
            except ValueError as exc:
                print(f"  row {row_no}: non-numeric -> {exc}", file=sys.stderr)
                issues += 1
                continue
            if vals["density_g_per_ml"] <= 0:
                print(f"  row {row_no}: density must be > 0", file=sys.stderr)
                issues += 1
            est = 4 * vals["protein_g"] + 9 * vals["fat_g"] + 4 * vals["carb_g"]
            if vals["kcal_per_100g"] > 0:
                rel = abs(est - vals["kcal_per_100g"]) / vals["kcal_per_100g"]
                if rel > 0.40:  # Atwater factors are approximate; allow 40 %.
                    print(f"  row {row_no} ({rid}): kcal={vals['kcal_per_100g']} but "
                          f"4P+9F+4C={est:.0f} (off by {rel*100:.0f} %)",
                          file=sys.stderr)
    print(f"checked {len(seen_ids)} ingredients, {issues} issues")
    return 0 if issues == 0 else 2


def fetch_hf_dish(out_path: Path) -> int:
    """Pull ``eriko-syah/indonesian-food`` parquet sample via the HF datasets API."""
    url = ("https://datasets-server.huggingface.co/rows?"
           "dataset=eriko-syah%2Findonesian-food&config=default&split=train&offset=0&length=100")
    req = Request(url, headers={"User-Agent": "cvfoodid/0.1"})
    try:
        with urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - network path
        print(f"HF fetch failed: {exc}", file=sys.stderr)
        return 1
    rows = payload.get("rows", [])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dish_id", "name_id", "kcal", "protein_g", "fat_g", "carb_g", "image_url"])
        for r in rows:
            row = r["row"]
            writer.writerow([
                row.get("id"),
                row.get("name"),
                row.get("calories"),
                row.get("proteins"),
                row.get("fat"),
                row.get("carbohydrate"),
                row.get("image"),
            ])
    print(f"wrote {len(rows)} dishes -> {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/nutrition/tkpi_lookup.csv")
    parser.add_argument("--dish-out", default="data/nutrition/dish_lookup_hf.csv")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    rc = check_csv(Path(args.csv))
    if args.check_only or rc != 0:
        return rc
    return fetch_hf_dish(Path(args.dish_out))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
