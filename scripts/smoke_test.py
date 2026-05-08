"""End-to-end smoke test that does NOT require a trained YOLO model.

Builds a synthetic detection list, walks it through the portion estimator and
nutrition calculator, and asserts the totals are sensible. Useful as a CI
sanity check while the full model trains separately.

Run from the repo root::

    python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import sys

from cvfoodid.nutrition.calculator import IngredientPortion, NutritionCalculator
from cvfoodid.nutrition.database import NutritionDatabase
from cvfoodid.portion.estimator import DetectionInput, MassEstimator


def main() -> int:
    db = NutritionDatabase()
    calc = NutritionCalculator(db)
    estimator = MassEstimator(db)

    # Synthetic "nasi goreng" plate: rice + egg + chicken on a 24 cm plate.
    # Image is 1080 px wide; plate fills ~960 px -> mm_per_pixel = 240/960 = 0.25.
    mm_per_pixel = 240.0 / 960.0
    detections = [
        DetectionInput("nasi_putih", (200, 200, 800, 800), 0.92),
        DetectionInput("telur_dadar", (650, 500, 850, 700), 0.88),
        DetectionInput("daging_ayam", (250, 600, 500, 800), 0.81),
    ]
    portions = []
    for d in detections:
        mass = estimator.estimate(d, mm_per_pixel)
        portions.append(IngredientPortion(
            ingredient_id=d.ingredient_id,
            mass_g=mass,
            confidence=d.confidence,
            bbox_xyxy=d.bbox_xyxy,
        ))
    result = calc.compute(portions)

    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))

    assert result.total_mass_g > 50, f"unexpectedly small mass: {result.total_mass_g}"
    assert result.total_kcal > 50, f"unexpectedly small kcal: {result.total_kcal}"
    assert len(result.lines) == len(detections)
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
