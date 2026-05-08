"""Tests for the bilingual label helpers."""

from __future__ import annotations

from cvfoodid.i18n import Lang, label


def test_indonesian_translation() -> None:
    assert "Kalori" in label("kcal", Lang.ID)


def test_english_translation() -> None:
    assert "Calories" in label("kcal", Lang.EN)


def test_unknown_key_falls_back_to_key() -> None:
    assert label("not_a_real_key", Lang.ID) == "not_a_real_key"
