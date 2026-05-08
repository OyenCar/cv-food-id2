"""Thin wrapper around Ultralytics YOLO for ingredient detection.

The wrapper isolates the rest of the pipeline from the heavy ``ultralytics``
import so that lightweight callers (e.g. unit tests, the nutrition calculator)
can run without it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """One YOLO detection mapped back to the ingredient vocabulary."""

    class_id: int
    ingredient_id: str  # YOLO class name; matches NutritionDatabase id
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    mask_pixels: int | None = None


class IngredientDetector:
    """Run YOLOv8/v11 on an image and return :class:`Detection` records.

    The Ultralytics import is lazy so importing this module does not require
    PyTorch to be installed.
    """

    def __init__(self, weights_path: str | Path,
                 conf: float = 0.25,
                 iou: float = 0.45,
                 device: str | None = None) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")
        self.conf = conf
        self.iou = iou
        self.device = device
        self._model = None  # lazy

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            ultralytics = importlib.import_module("ultralytics")
        except ImportError as exc:  # pragma: no cover - exercised only in env
            raise ImportError(
                "Ultralytics is required for inference. "
                "Install with: pip install ultralytics"
            ) from exc
        self._model = ultralytics.YOLO(str(self.weights_path))

    def predict(self, image: np.ndarray | str | Path) -> list[Detection]:
        """Return all detections in the image, mapped to ingredient IDs."""
        self._load()
        assert self._model is not None
        kwargs: dict[str, object] = {"conf": self.conf, "iou": self.iou, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self._model.predict(image, **kwargs)
        if not results:
            return []
        r = results[0]
        names = r.names if hasattr(r, "names") else self._model.names
        out: list[Detection] = []
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return out
        masks = getattr(r, "masks", None)
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            mask_pixels: int | None = None
            if masks is not None and masks.data is not None and i < len(masks.data):
                m = masks.data[i].cpu().numpy()
                mask_pixels = int((m > 0).sum())
            out.append(
                Detection(
                    class_id=cls_id,
                    ingredient_id=str(names[cls_id]),
                    confidence=conf,
                    bbox_xyxy=tuple(xyxy),  # type: ignore[arg-type]
                    mask_pixels=mask_pixels,
                )
            )
        return out
