"""Bilingual (Indonesian / English) label helpers."""

from __future__ import annotations

from enum import Enum


class Lang(str, Enum):
    ID = "id"
    EN = "en"


# UI strings used by the demo and CLI. Keep small and stable.
_STRINGS: dict[str, dict[Lang, str]] = {
    "title": {
        Lang.ID: "Deteksi Kandungan Makanan Indonesia",
        Lang.EN: "Indonesian Food Content Detector",
    },
    "ingredient": {Lang.ID: "Bahan", Lang.EN: "Ingredient"},
    "mass_g": {Lang.ID: "Massa (g)", Lang.EN: "Mass (g)"},
    "kcal": {Lang.ID: "Kalori (kkal)", Lang.EN: "Calories (kcal)"},
    "protein": {Lang.ID: "Protein (g)", Lang.EN: "Protein (g)"},
    "fat": {Lang.ID: "Lemak (g)", Lang.EN: "Fat (g)"},
    "carb": {Lang.ID: "Karbohidrat (g)", Lang.EN: "Carbohydrate (g)"},
    "fiber": {Lang.ID: "Serat (g)", Lang.EN: "Fiber (g)"},
    "total": {Lang.ID: "Total", Lang.EN: "Total"},
    "no_food_detected": {
        Lang.ID: "Tidak ada bahan makanan terdeteksi.",
        Lang.EN: "No food ingredient detected.",
    },
}


def label(key: str, lang: Lang = Lang.ID) -> str:
    """Return the localized UI string for `key` in `lang`."""
    if key not in _STRINGS:
        return key
    return _STRINGS[key].get(lang, _STRINGS[key][Lang.EN])
