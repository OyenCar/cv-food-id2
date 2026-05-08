"""Tests for portion estimation and reference-object scaling."""

from __future__ import annotations

import numpy as np
import pytest

from cvfoodid.portion.estimator import DetectionInput, MassEstimator
from cvfoodid.portion.reference import (
    KNOWN_OBJECTS_MM,
    from_image_size_heuristic,
    from_known_object,
    from_plate_diameter,
)


def test_known_object_coin_scales_correctly() -> None:
    # IDR 1000 coin is 24.5 mm; if its bbox is 49 px wide, mm/px should be 0.5.
    ref = from_known_object((100, 100, 149, 149), "coin_idr_1000")
    assert ref.mm_per_pixel == pytest.approx(0.5, rel=1e-6)
    assert ref.confidence > 0.9


def test_known_object_unknown_class_raises() -> None:
    with pytest.raises(ValueError):
        from_known_object((0, 0, 10, 10), "not_a_real_object")


def test_known_object_dimensions_complete() -> None:
    # Every reference class declared by the pipeline must have a real-world size.
    expected = {"coin_idr_100", "coin_idr_200", "coin_idr_500", "coin_idr_1000",
                "ktp", "spoon_table", "chopstick"}
    assert expected.issubset(KNOWN_OBJECTS_MM.keys())


def test_image_size_heuristic_is_reasonable() -> None:
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ref = from_image_size_heuristic(img)
    assert 0.5 < ref.mm_per_pixel < 5
    assert ref.confidence < 0.5  # low confidence by design


def test_plate_diameter_returns_none_when_no_plate() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ref = from_plate_diameter(img, plate_diameter_mm=240.0)
    # Empty image has no contours -> None expected.
    assert ref is None


def test_estimator_basic_mass() -> None:
    est = MassEstimator()
    det = DetectionInput("nasi_putih", (0, 0, 100, 100), 0.9)
    # 100x100 px bbox at 1 mm/px = 100x100 mm -> 25 mm tall pile of rice.
    mass = est.estimate(det, mm_per_pixel=1.0)
    # Volume = 100*100 * pi/4 * 28 = ~219 911 mm^3 = ~219.9 ml * 0.85 = ~187 g.
    assert 100 < mass < 300, mass


def test_estimator_scales_with_area_squared() -> None:
    est = MassEstimator()
    small = est.estimate(DetectionInput("nasi_putih", (0, 0, 100, 100), 1.0), mm_per_pixel=1.0)
    big = est.estimate(DetectionInput("nasi_putih", (0, 0, 200, 200), 1.0), mm_per_pixel=1.0)
    # Doubling each side should ~quadruple the mass (area * height).
    assert big == pytest.approx(4 * small, rel=1e-6)


def test_estimator_rejects_invalid_scale() -> None:
    est = MassEstimator()
    with pytest.raises(ValueError):
        est.estimate(DetectionInput("nasi_putih", (0, 0, 10, 10), 1.0), mm_per_pixel=0.0)


def test_estimator_unknown_ingredient_uses_fallback() -> None:
    est = MassEstimator()
    mass = est.estimate(DetectionInput("not_in_db", (0, 0, 50, 50), 1.0), mm_per_pixel=1.0)
    # No record -> density=1, height=15 mm -> ~ 50 * 50 * pi/4 * 15 / 1000 = ~29 g.
    assert mass > 0
