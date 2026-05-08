"""End-to-end inference pipeline.

Image -> ingredient detections -> mass estimation -> nutrition totals.

Designed to be importable without ``ultralytics`` so that the calculator and
estimator pieces can be exercised independently in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cvfoodid.nutrition.calculator import (
    IngredientPortion,
    NutritionCalculator,
    NutritionResult,
)
from cvfoodid.nutrition.database import NutritionDatabase
from cvfoodid.portion.estimator import DetectionInput, MassEstimator
from cvfoodid.portion.reference import (
    ReferenceMeasurement,
    from_image_size_heuristic,
    from_known_object,
    from_plate_diameter,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cvfoodid.detection.detector import Detection, IngredientDetector


# Ingredient class names that are *reference objects*, not food. Detections
# matching these are used for scale recovery and excluded from the meal.
REFERENCE_CLASSES: frozenset[str] = frozenset(
    {"coin_idr_100", "coin_idr_200", "coin_idr_500", "coin_idr_1000",
     "ktp", "spoon_table", "chopstick"}
)


@dataclass(slots=True)
class PipelineOutput:
    detections: list[Detection]
    reference: ReferenceMeasurement
    nutrition: NutritionResult

    def as_dict(self) -> dict[str, object]:
        return {
            "reference": {
                "method": self.reference.method,
                "mm_per_pixel": round(self.reference.mm_per_pixel, 4),
                "confidence": round(self.reference.confidence, 3),
                "notes": self.reference.notes,
            },
            **self.nutrition.as_dict(),
        }


class FoodPipeline:
    """Glue layer that runs detection + portion estimation + nutrition lookup."""

    def __init__(self,
                 detector: IngredientDetector,
                 db: NutritionDatabase | None = None,
                 plate_diameter_mm: float | None = 240.0) -> None:
        """Parameters
        ----------
        detector : IngredientDetector
            A loaded YOLO detector.
        db : NutritionDatabase, optional
            Custom nutrition database. Defaults to the bundled TKPI CSV.
        plate_diameter_mm : float, optional
            Fallback plate size for scale recovery when no reference object
            is detected. Set to ``None`` to disable plate-based scaling.
        """
        self.detector = detector
        self.db = db or NutritionDatabase()
        self.calculator = NutritionCalculator(self.db)
        self.estimator = MassEstimator(self.db)
        self.plate_diameter_mm = plate_diameter_mm

    def _recover_scale(self, image_bgr: np.ndarray, detections: list[Detection]) -> ReferenceMeasurement:
        # 1. Prefer a known reference object detected by YOLO.
        for d in detections:
            if d.ingredient_id in REFERENCE_CLASSES:
                return from_known_object(d.bbox_xyxy, d.ingredient_id)
        # 2. Fall back to plate-diameter ellipse fitting.
        if self.plate_diameter_mm is not None:
            ref = from_plate_diameter(image_bgr, self.plate_diameter_mm)
            if ref is not None:
                return ref
        # 3. Last-resort heuristic.
        return from_image_size_heuristic(image_bgr)

    def run(self, image_path: str | Path) -> PipelineOutput:
        import cv2

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        detections = self.detector.predict(image_bgr)
        reference = self._recover_scale(image_bgr, detections)
        portions: list[IngredientPortion] = []
        for d in detections:
            if d.ingredient_id in REFERENCE_CLASSES:
                continue
            det_input = DetectionInput(
                ingredient_id=d.ingredient_id,
                bbox_xyxy=d.bbox_xyxy,
                confidence=d.confidence,
                mask_pixels=d.mask_pixels,
            )
            mass_g = self.estimator.estimate(det_input, reference.mm_per_pixel)
            portions.append(
                IngredientPortion(
                    ingredient_id=d.ingredient_id,
                    mass_g=mass_g,
                    confidence=d.confidence,
                    bbox_xyxy=d.bbox_xyxy,
                )
            )
        nutrition = self.calculator.compute(portions)
        return PipelineOutput(detections=detections, reference=reference, nutrition=nutrition)
