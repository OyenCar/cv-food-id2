"""Aggregate ingredient masses into a per-plate nutrition summary."""

from __future__ import annotations

from dataclasses import dataclass, field

from cvfoodid.nutrition.database import IngredientRecord, NutritionDatabase


@dataclass(frozen=True, slots=True)
class IngredientPortion:
    """One detected ingredient with its estimated mass in grams."""

    ingredient_id: str
    mass_g: float
    confidence: float = 1.0
    bbox_xyxy: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class IngredientLine:
    record: IngredientRecord
    mass_g: float
    confidence: float
    kcal: float
    protein_g: float
    fat_g: float
    carb_g: float
    fiber_g: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "id": self.record.id,
            "name_id": self.record.name_id,
            "name_en": self.record.name_en,
            "mass_g": round(self.mass_g, 2),
            "confidence": round(self.confidence, 3),
            "kcal": round(self.kcal, 1),
            "protein_g": round(self.protein_g, 2),
            "fat_g": round(self.fat_g, 2),
            "carb_g": round(self.carb_g, 2),
            "fiber_g": round(self.fiber_g, 2),
        }


@dataclass(slots=True)
class NutritionResult:
    lines: list[IngredientLine] = field(default_factory=list)
    total_kcal: float = 0.0
    total_protein_g: float = 0.0
    total_fat_g: float = 0.0
    total_carb_g: float = 0.0
    total_fiber_g: float = 0.0
    total_mass_g: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "ingredients": [line.as_dict() for line in self.lines],
            "total": {
                "mass_g": round(self.total_mass_g, 2),
                "kcal": round(self.total_kcal, 1),
                "protein_g": round(self.total_protein_g, 2),
                "fat_g": round(self.total_fat_g, 2),
                "carb_g": round(self.total_carb_g, 2),
                "fiber_g": round(self.total_fiber_g, 2),
            },
        }


class NutritionCalculator:
    """Compute total nutrition from a list of detected portions."""

    def __init__(self, db: NutritionDatabase | None = None) -> None:
        self.db = db or NutritionDatabase()

    def compute(self, portions: list[IngredientPortion]) -> NutritionResult:
        result = NutritionResult()
        for p in portions:
            rec = self.db.get(p.ingredient_id)
            if rec is None:
                # Unknown ingredient -> skip silently to keep pipeline robust.
                # The caller can audit by comparing input vs output line count.
                continue
            scaled = rec.scaled(p.mass_g)
            line = IngredientLine(
                record=rec,
                mass_g=p.mass_g,
                confidence=p.confidence,
                kcal=scaled["kcal"],
                protein_g=scaled["protein_g"],
                fat_g=scaled["fat_g"],
                carb_g=scaled["carb_g"],
                fiber_g=scaled["fiber_g"],
            )
            result.lines.append(line)
            result.total_kcal += line.kcal
            result.total_protein_g += line.protein_g
            result.total_fat_g += line.fat_g
            result.total_carb_g += line.carb_g
            result.total_fiber_g += line.fiber_g
            result.total_mass_g += line.mass_g
        return result
