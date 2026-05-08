"""Tests for NutritionCalculator aggregation logic."""

from __future__ import annotations

import pytest

from cvfoodid.nutrition.calculator import IngredientPortion, NutritionCalculator
from cvfoodid.nutrition.database import NutritionDatabase


@pytest.fixture(scope="module")
def calc() -> NutritionCalculator:
    return NutritionCalculator(NutritionDatabase())


def test_empty_input_yields_zero_totals(calc: NutritionCalculator) -> None:
    out = calc.compute([])
    assert out.total_kcal == 0
    assert out.total_protein_g == 0
    assert out.lines == []


def test_single_ingredient(calc: NutritionCalculator) -> None:
    out = calc.compute([IngredientPortion("nasi_putih", 200.0, 0.9)])
    assert out.total_mass_g == pytest.approx(200.0)
    # Nasi putih = 130 kcal/100g -> 200 g = 260 kcal.
    assert out.total_kcal == pytest.approx(260.0, rel=1e-6)
    assert len(out.lines) == 1


def test_multi_ingredient_totals_sum(calc: NutritionCalculator) -> None:
    portions = [
        IngredientPortion("nasi_putih", 200.0, 0.9),
        IngredientPortion("daging_ayam", 80.0, 0.8),
        IngredientPortion("sayur_kangkung", 50.0, 0.7),
    ]
    out = calc.compute(portions)
    expected_kcal = 130 * 2 + 239 * 0.8 + 19 * 0.5
    assert out.total_kcal == pytest.approx(expected_kcal, rel=1e-6)
    assert out.total_mass_g == pytest.approx(330.0)
    assert len(out.lines) == 3


def test_unknown_ingredient_skipped_silently(calc: NutritionCalculator) -> None:
    out = calc.compute([
        IngredientPortion("nasi_putih", 100.0, 0.9),
        IngredientPortion("does_not_exist", 200.0, 0.1),
    ])
    assert len(out.lines) == 1
    assert out.lines[0].record.id == "nasi_putih"


def test_as_dict_roundtrips(calc: NutritionCalculator) -> None:
    out = calc.compute([IngredientPortion("tempe", 50.0)])
    d = out.as_dict()
    assert "total" in d
    assert "ingredients" in d
    assert d["ingredients"][0]["id"] == "tempe"  # type: ignore[index]
