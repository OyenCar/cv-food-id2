"""Wire-level smoke test of the calculator + estimator path.

This intentionally avoids importing the YOLO detector so the test suite stays
fast and runs without ``ultralytics`` installed.
"""

from __future__ import annotations

from cvfoodid.nutrition.calculator import IngredientPortion, NutritionCalculator
from cvfoodid.nutrition.database import NutritionDatabase
from cvfoodid.portion.estimator import DetectionInput, MassEstimator


def test_synthetic_nasi_goreng_plate_is_sane() -> None:
    db = NutritionDatabase()
    estimator = MassEstimator(db)
    calc = NutritionCalculator(db)

    mm_per_pixel = 240.0 / 960.0  # 24 cm plate fills 960 px.
    detections = [
        DetectionInput("nasi_putih", (200, 200, 800, 800), 0.92),
        DetectionInput("telur_dadar", (650, 500, 850, 700), 0.88),
        DetectionInput("daging_ayam", (250, 600, 500, 800), 0.81),
    ]
    portions = [
        IngredientPortion(d.ingredient_id, estimator.estimate(d, mm_per_pixel),
                          d.confidence, d.bbox_xyxy)
        for d in detections
    ]
    result = calc.compute(portions)
    assert len(result.lines) == 3
    assert 100 < result.total_mass_g < 1500
    assert 100 < result.total_kcal < 4000
    assert result.total_protein_g > 0
    assert result.total_carb_g > 0
