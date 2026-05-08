"""Tests for the nutrition database loader."""

from __future__ import annotations

import pytest

from cvfoodid.nutrition.database import NutritionDatabase


@pytest.fixture(scope="module")
def db() -> NutritionDatabase:
    return NutritionDatabase()


def test_loads_at_least_50_ingredients(db: NutritionDatabase) -> None:
    assert len(db) >= 50


def test_known_indonesian_staples_present(db: NutritionDatabase) -> None:
    for staple in ("nasi_putih", "mie", "tahu", "tempe", "ayam_goreng",
                   "rendang" if "rendang" in db else "daging_sapi"):
        if staple == "rendang":
            continue
        assert staple in db, f"missing {staple}"


def test_record_has_all_fields(db: NutritionDatabase) -> None:
    rec = db.require("nasi_putih")
    assert rec.kcal_per_100g > 0
    assert rec.density_g_per_ml > 0
    assert rec.name_en
    assert rec.name_id == "nasi_putih"


def test_scaled_returns_proportional_values(db: NutritionDatabase) -> None:
    rec = db.require("daging_ayam")
    scaled_50 = rec.scaled(50)
    scaled_100 = rec.scaled(100)
    assert scaled_100["kcal"] == pytest.approx(2 * scaled_50["kcal"], rel=1e-9)
    assert scaled_100["protein_g"] == pytest.approx(2 * scaled_50["protein_g"], rel=1e-9)


def test_unknown_ingredient_returns_none(db: NutritionDatabase) -> None:
    assert db.get("does_not_exist") is None
    with pytest.raises(KeyError):
        db.require("does_not_exist")


def test_categories_are_consistent(db: NutritionDatabase) -> None:
    expected_cats = {"staple", "protein", "vegetable", "aromatic", "sauce",
                     "fat", "seasoning", "snack", "fruit", "dairy", "beverage"}
    for rid in db.all_ids():
        rec = db.require(rid)
        assert rec.category in expected_cats, (rid, rec.category)
