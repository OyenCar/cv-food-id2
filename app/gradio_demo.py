"""Gradio demo for cvfoodid.

Loads a YOLO checkpoint and lets the user upload a food image to see the
detected ingredients, estimated masses, and total nutrition.

Run locally::

    python app/gradio_demo.py --weights runs/detect/cvfoodid-yolo/weights/best.pt

Or set ``CVFOODID_WEIGHTS`` env var and just::

    python app/gradio_demo.py
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from cvfoodid.i18n import Lang, label

if TYPE_CHECKING:  # pragma: no cover
    from cvfoodid.pipeline import FoodPipeline


def _annotate(image_bgr: np.ndarray, detections, reference) -> np.ndarray:
    out = image_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        text = f"{d.ingredient_id} {d.confidence:.2f}"
        cv2.putText(out, text, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"scale={reference.mm_per_pixel:.3f} mm/px ({reference.method})",
                (8, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 200), 1, cv2.LINE_AA)
    return out


def make_app(pipeline: FoodPipeline):
    import gradio as gr

    def infer(image: np.ndarray | None, lang_choice: str, plate_mm: float):
        if image is None:
            return None, label("no_food_detected", Lang(lang_choice)), {}
        # Gradio gives us RGB; pipeline expects BGR via cv2.imread, so we go
        # through a tempfile to keep the code path uniform with the CLI.
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, bgr)
            pipeline.plate_diameter_mm = plate_mm or pipeline.plate_diameter_mm
            out = pipeline.run(tmp.name)
        annotated = _annotate(bgr, out.detections, out.reference)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        lang = Lang(lang_choice)
        result_dict = out.as_dict()
        total = result_dict["total"]  # type: ignore[index]
        summary = (
            f"**{label('total', lang)}**\n\n"
            f"- {label('mass_g', lang)}: **{total['mass_g']}**\n"  # type: ignore[index]
            f"- {label('kcal', lang)}: **{total['kcal']}**\n"      # type: ignore[index]
            f"- {label('protein', lang)}: {total['protein_g']}\n"  # type: ignore[index]
            f"- {label('fat', lang)}: {total['fat_g']}\n"          # type: ignore[index]
            f"- {label('carb', lang)}: {total['carb_g']}\n"        # type: ignore[index]
        )
        return annotated_rgb, summary, result_dict

    with gr.Blocks(title=label("title", Lang.ID)) as app:
        gr.Markdown(f"# {label('title', Lang.ID)} / {label('title', Lang.EN)}")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="numpy", label="Foto makanan / Food photo")
                lang_choice = gr.Radio(["id", "en"], value="id", label="Language")
                plate_mm = gr.Number(value=240.0, label="Plate diameter (mm)")
                btn = gr.Button("Hitung gizi / Compute nutrition", variant="primary")
            with gr.Column():
                out_image = gr.Image(label="Detections")
                out_text = gr.Markdown()
                out_json = gr.JSON(label="Raw result")
        btn.click(infer, inputs=[inp, lang_choice, plate_mm],
                  outputs=[out_image, out_text, out_json])
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=os.getenv("CVFOODID_WEIGHTS"))
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--plate-mm", type=float, default=240.0)
    args = parser.parse_args()

    if not args.weights:
        raise SystemExit(
            "No YOLO weights provided. Pass --weights or set CVFOODID_WEIGHTS env var."
        )
    if not Path(args.weights).is_file():
        raise SystemExit(f"Weights not found: {args.weights}")

    from cvfoodid.detection.detector import IngredientDetector
    from cvfoodid.pipeline import FoodPipeline

    detector = IngredientDetector(weights_path=args.weights, conf=0.25)
    pipeline = FoodPipeline(detector=detector, plate_diameter_mm=args.plate_mm)
    app = make_app(pipeline)
    app.launch(server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
