"""Indonesian nutrition database.

Loads the TKPI-derived CSV at ``data/nutrition/tkpi_lookup.csv`` and exposes
fast lookups by ingredient ID (which doubles as the YOLO class name).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class IngredientRecord:
    """A single row from ``tkpi_lookup.csv``.

    All nutrient values are per 100 g of edible portion.
    """

    id: str
    name_id: str
    name_en: str
    category: str
    kcal_per_100g: float
    protein_g: float
    fat_g: float
    carb_g: float
    fiber_g: float
    water_g: float
    density_g_per_ml: float
    source: str

    def scaled(self, mass_g: float) -> dict[str, float]:
        """Return nutrient amounts for ``mass_g`` grams of this ingredient."""
        factor = mass_g / 100.0
        return {
            "kcal": self.kcal_per_100g * factor,
            "protein_g": self.protein_g * factor,
            "fat_g": self.fat_g * factor,
            "carb_g": self.carb_g * factor,
            "fiber_g": self.fiber_g * factor,
        }


def _default_csv_path() -> Path:
    """Locate ``tkpi_lookup.csv`` relative to the repo root."""
    here = Path(__file__).resolve()
    # src/cvfoodid/nutrition/database.py -> repo root is parents[3]
    repo_root = here.parents[3]
    return repo_root / "data" / "nutrition" / "tkpi_lookup.csv"


class NutritionDatabase:
    """Read-only lookup of Indonesian ingredient nutrition values."""

    def __init__(self, csv_path: Path | str | None = None) -> None:
        self.csv_path = Path(csv_path) if csv_path else _default_csv_path()
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"Nutrition CSV not found at {self.csv_path}. "
                "Pass csv_path explicitly or run from the repo root."
            )
        self._records: dict[str, IngredientRecord] = {}
        self._load()

    def _load(self) -> None:
        with self.csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rec = IngredientRecord(
                    id=row["name_id"],
                    name_id=row["name_id"],
                    name_en=row["name_en"],
                    category=row["category"],
                    kcal_per_100g=float(row["kcal_per_100g"]),
                    protein_g=float(row["protein_g"]),
                    fat_g=float(row["fat_g"]),
                    carb_g=float(row["carb_g"]),
                    fiber_g=float(row["fiber_g"]),
                    water_g=float(row["water_g"]),
                    density_g_per_ml=float(row["density_g_per_ml"]),
                    source=row["source"],
                )
                self._records[rec.id] = rec

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, ingredient_id: object) -> bool:
        return isinstance(ingredient_id, str) and ingredient_id in self._records

    def get(self, ingredient_id: str) -> IngredientRecord | None:
        """Return the record for ``ingredient_id`` (the YOLO class name)."""
        return self._records.get(ingredient_id)

    def require(self, ingredient_id: str) -> IngredientRecord:
        """Like :meth:`get` but raise ``KeyError`` if missing."""
        rec = self.get(ingredient_id)
        if rec is None:
            raise KeyError(f"Unknown ingredient: {ingredient_id!r}")
        return rec

    def all_ids(self) -> list[str]:
        """All ingredient IDs in load order (matches CSV row order)."""
        return list(self._records.keys())

    def by_category(self, category: str) -> list[IngredientRecord]:
        return [r for r in self._records.values() if r.category == category]
