"""Reference-object detection for absolute scale recovery.

The challenge with single-image portion estimation is the lack of metric scale:
two photos of the same plate can look identical at different distances. We
solve this by detecting a *reference object* of known real-world size and
computing pixels-per-millimeter from it.

Three reference strategies are supported, in order of preference:
1. ``known_object``: a coin, ID card, spoon, etc. detected by the same YOLO
   model (with ``coin_idr_1000`` etc. as classes).
2. ``plate_diameter``: the user supplies plate diameter (mm); we fit an
   ellipse to the plate and use the major axis.
3. ``hand``: detect the user's hand for a coarse fallback (~ +-30% error).

All functions return ``mm_per_pixel`` (float). Higher = each pixel covers more
real-world space (camera farther away).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Real-world dimensions (mm) of common Indonesian reference objects.
# Sources: Bank Indonesia coin specs, KTP standard ISO/IEC 7810 ID-1.
KNOWN_OBJECTS_MM: dict[str, float] = {
    "coin_idr_100": 23.0,    # diameter
    "coin_idr_200": 25.0,
    "coin_idr_500": 27.0,
    "coin_idr_1000": 24.5,
    "ktp": 85.6,             # long edge of ID-1 card
    "spoon_table": 175.0,    # length of an Indonesian dessert spoon (typical)
    "chopstick": 230.0,
}


@dataclass(frozen=True, slots=True)
class ReferenceMeasurement:
    """Result of a reference-object measurement."""

    method: str
    mm_per_pixel: float
    confidence: float
    notes: str = ""


def from_known_object(bbox_xyxy: tuple[float, float, float, float], object_class: str) -> ReferenceMeasurement:
    """Compute scale from a detected known object's bbox.

    Uses the bbox's longer side as the proxy for the object's longest real
    dimension. For round objects (coin) both sides are equal so it does not
    matter; for rectangular objects (KTP) the longer side is the long edge.
    """
    if object_class not in KNOWN_OBJECTS_MM:
        raise ValueError(
            f"Unknown reference object {object_class!r}. "
            f"Add it to KNOWN_OBJECTS_MM with its real-world size."
        )
    real_mm = KNOWN_OBJECTS_MM[object_class]
    x1, y1, x2, y2 = bbox_xyxy
    pixel_size = max(x2 - x1, y2 - y1)
    if pixel_size <= 0:
        raise ValueError(f"Degenerate bbox: {bbox_xyxy}")
    mm_per_pixel = real_mm / pixel_size
    return ReferenceMeasurement(
        method=f"known_object:{object_class}",
        mm_per_pixel=mm_per_pixel,
        confidence=0.95,
        notes=f"{object_class}={real_mm}mm, pixels={pixel_size:.1f}",
    )


def fit_plate_ellipse(image_bgr: np.ndarray) -> tuple[float, float, tuple[float, float], float] | None:
    """Detect the plate as an ellipse and return ``(cx, cy, (a, b), angle)``.

    The plate is found by looking for the largest near-circular contour after
    converting to grayscale and applying adaptive thresholding. ``a`` and ``b``
    are semi-axes in pixels. Returns ``None`` if no plate found.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = gray.shape[:2]
    image_area = float(h * w)
    candidates: list[tuple[float, tuple]] = []
    for c in contours:
        if len(c) < 5:
            continue
        area = cv2.contourArea(c)
        if area < image_area * 0.05 or area > image_area * 0.95:
            continue
        ellipse = cv2.fitEllipse(c)
        (_cx, _cy), (axis_a, axis_b), _angle = ellipse
        if axis_a <= 0 or axis_b <= 0:
            continue
        # Roundness: ratio of minor to major; plates are mostly circular when
        # viewed from above so this should be close to 1.
        roundness = min(axis_a, axis_b) / max(axis_a, axis_b)
        if roundness < 0.4:
            continue
        candidates.append((area * roundness, ellipse))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    (cx, cy), (axis_a, axis_b), angle = candidates[0][1]
    return (float(cx), float(cy), (float(axis_a / 2.0), float(axis_b / 2.0)), float(angle))


def from_plate_diameter(image_bgr: np.ndarray, plate_diameter_mm: float) -> ReferenceMeasurement | None:
    """Compute scale by fitting an ellipse to the plate.

    The major axis (in pixels) corresponds to ``plate_diameter_mm``.
    """
    fit = fit_plate_ellipse(image_bgr)
    if fit is None:
        return None
    _, _, (a, b), _ = fit
    major_pixels = 2.0 * max(a, b)
    if major_pixels <= 0:
        return None
    return ReferenceMeasurement(
        method="plate_diameter",
        mm_per_pixel=plate_diameter_mm / major_pixels,
        confidence=0.75,
        notes=f"plate_d={plate_diameter_mm}mm, major_px={major_pixels:.1f}",
    )


def from_image_size_heuristic(image_bgr: np.ndarray) -> ReferenceMeasurement:
    """Fallback when no reference is available.

    Assumes the photo was taken from typical phone-eating distance (~30 cm)
    with a standard ~26mm equivalent lens, which puts roughly 2 mm per pixel
    on a 1080p image. This is *very* approximate (+- 50%) and should only be
    used as a degraded fallback.
    """
    h, w = image_bgr.shape[:2]
    long_side = float(max(h, w))
    # 1080-px long side -> ~ 2 mm/px assumption; scale linearly otherwise.
    mm_per_pixel = 2.0 * (1080.0 / long_side)
    return ReferenceMeasurement(
        method="image_size_heuristic",
        mm_per_pixel=mm_per_pixel,
        confidence=0.30,
        notes="No reference object detected; fallback heuristic.",
    )
