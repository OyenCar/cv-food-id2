"""Portion / mass estimation from detected ingredients.

Given a list of detected ingredient bboxes (or masks) and an absolute scale
(``mm_per_pixel``), estimate the mass in grams of each ingredient using:

    mass_g = projected_area_mm2 * effective_height_mm * density_g_per_ml

Where ``effective_height_mm`` is a per-category empirical pile height (e.g.
rice piles ~25 mm tall on average; sauces ~5 mm; meat slabs ~12 mm).

This is a deliberately simple geometric model. For higher accuracy, use
RGB-D depth (Nutrition5k-style) or a learned regression head -- both are
left as TODOs in :class:`MassEstimator`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cvfoodid.nutrition.database import NutritionDatabase

# Empirical pile heights (mm) by ingredient category. Values calibrated
# against typical Indonesian plate photos and food-packaging guides.
DEFAULT_HEIGHTS_MM: dict[str, float] = {
    "staple": 25.0,      # rice mounds, noodles
    "protein": 18.0,     # meat / fish slabs, fried chicken
    "vegetable": 15.0,   # sayur piles
    "aromatic": 5.0,     # bawang, cabai garnish
    "sauce": 4.0,        # kuah / saus pools
    "fat": 3.0,
    "seasoning": 2.0,
    "fruit": 22.0,
    "snack": 8.0,
    "dairy": 8.0,
    "beverage": 30.0,    # treat as a column of liquid
}

# Per-ingredient overrides (when a single category default is too coarse).
INGREDIENT_HEIGHT_OVERRIDES_MM: dict[str, float] = {
    "telur_ayam": 20.0,
    "telur_dadar": 8.0,
    "ayam_goreng": 25.0,
    "daging_sapi": 15.0,
    "tahu": 20.0,
    "tempe": 12.0,
    "kerupuk": 4.0,
    "santan": 6.0,
    "nasi_putih": 28.0,
    "nasi_kuning": 28.0,
    "nasi_uduk": 28.0,
    "mie": 22.0,
    "bihun": 18.0,
    "bakso": 25.0,
    "saus_kacang": 6.0,
}


@dataclass(frozen=True, slots=True)
class DetectionInput:
    """A YOLO-style detection ready for mass estimation."""

    ingredient_id: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    mask_pixels: int | None = None  # set if instance segmentation is used

    def projected_area_pixels(self) -> float:
        """Return the projected (top-down) area in pixel^2.

        Uses the segmentation mask area when available; otherwise falls back
        to the bbox area. The bbox fallback overestimates by ~25% for round
        food piles, which is roughly compensated by the conservative pile
        heights.
        """
        if self.mask_pixels is not None and self.mask_pixels > 0:
            return float(self.mask_pixels)
        x1, y1, x2, y2 = self.bbox_xyxy
        # Round food on a plate occupies ~pi/4 of its bounding box.
        return max(0.0, (x2 - x1) * (y2 - y1)) * (np.pi / 4.0)


class MassEstimator:
    """Convert detections + scale into per-ingredient masses (grams).

    Examples
    --------
    >>> est = MassEstimator()
    >>> det = DetectionInput("nasi_putih", (100, 100, 400, 400), 0.9)
    >>> mass_g = est.estimate(det, mm_per_pixel=0.5)
    >>> mass_g > 0
    True
    """

    def __init__(self, db: NutritionDatabase | None = None,
                 heights_mm: dict[str, float] | None = None,
                 overrides_mm: dict[str, float] | None = None) -> None:
        self.db = db or NutritionDatabase()
        self.heights_mm = dict(DEFAULT_HEIGHTS_MM)
        if heights_mm:
            self.heights_mm.update(heights_mm)
        self.overrides_mm = dict(INGREDIENT_HEIGHT_OVERRIDES_MM)
        if overrides_mm:
            self.overrides_mm.update(overrides_mm)

    def _height_mm(self, ingredient_id: str) -> float:
        if ingredient_id in self.overrides_mm:
            return self.overrides_mm[ingredient_id]
        rec = self.db.get(ingredient_id)
        if rec is None:
            # Unknown ingredient -> use a middle-of-the-road default.
            return 15.0
        return self.heights_mm.get(rec.category, 15.0)

    def estimate(self, det: DetectionInput, mm_per_pixel: float) -> float:
        """Return the estimated mass in grams for ``det``.

        Parameters
        ----------
        det : DetectionInput
            One detected ingredient.
        mm_per_pixel : float
            Absolute scale recovered from a reference object.
        """
        if mm_per_pixel <= 0:
            raise ValueError(f"mm_per_pixel must be positive, got {mm_per_pixel}")
        area_px2 = det.projected_area_pixels()
        if area_px2 <= 0:
            return 0.0
        area_mm2 = area_px2 * (mm_per_pixel ** 2)
        height_mm = self._height_mm(det.ingredient_id)
        volume_mm3 = area_mm2 * height_mm
        volume_ml = volume_mm3 / 1000.0
        rec = self.db.get(det.ingredient_id)
        density = rec.density_g_per_ml if rec is not None else 1.0
        return float(volume_ml * density)

    def estimate_batch(self, detections: list[DetectionInput], mm_per_pixel: float) -> list[float]:
        return [self.estimate(d, mm_per_pixel) for d in detections]
