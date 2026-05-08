"""cvfoodid: Hybrid CV pipeline for Indonesian food.

Stages:
    1. Ingredient detection (YOLOv8/v11)
    2. Portion / mass estimation (reference object + plate ellipse)
    3. Nutrition lookup (TKPI / USDA)
"""

__version__ = "0.1.0"

from cvfoodid.i18n import Lang, label
from cvfoodid.nutrition.calculator import IngredientPortion, NutritionCalculator, NutritionResult
from cvfoodid.nutrition.database import NutritionDatabase

__all__ = [
    "NutritionDatabase",
    "NutritionCalculator",
    "IngredientPortion",
    "NutritionResult",
    "Lang",
    "label",
]
