# Architecture

## High-level flow

```
              +------------------+
   image ---->| 1. YOLO detector |---+
              +------------------+   |
                                     |  detections + reference object
                                     v
                            +-------------------+
                            | 2. Mass estimator |
                            +-------------------+
                                     |
                                     v
                            +-------------------+
                            | 3. Nutrition calc |
                            +-------------------+
                                     |
                                     v
                              {kcal, P, F, C, ...}
```

## Components

### `cvfoodid.detection`
Thin wrapper around Ultralytics YOLO so the rest of the codebase can stay
import-light. The detector returns `Detection` records keyed by ingredient ID
(matching `tkpi_lookup.csv`).

### `cvfoodid.portion`
Two responsibilities:

1. **Reference recovery** -- choose the best available signal to convert
   pixels to millimetres:
   - Detected reference object (coin, KTP, spoon).
   - Plate ellipse fit (default 24 cm).
   - Image-size heuristic (last resort).
2. **Mass estimation** -- per-detection geometric model:
   `mass_g = projected_area_mm2 * effective_height_mm * density_g_per_ml / 1000`

### `cvfoodid.nutrition`
Loads `tkpi_lookup.csv` once, then aggregates `IngredientPortion` records into
a `NutritionResult` summary.

### `cvfoodid.pipeline`
Glue layer with one method (`FoodPipeline.run`) that orchestrates the three
stages above. Reference-class detections (e.g. `coin_idr_500`) are *consumed*
for scaling and *excluded* from the nutrition output.

## Why a hybrid pipeline?

A single regression head trained directly on calorie estimation
(Nutrition5k-style) is simpler but:

- Requires per-ingredient mass labels, which are scarce for Indonesian food.
- Becomes a black box that can't surface the bahan list to the user.
- Doesn't compose with future improvements (e.g. swapping in a depth model).

The hybrid layout means:

- Detection improvements (new YOLO release, better data) propagate immediately.
- Portion estimation can be upgraded to RGB-D / learned regression without
  touching detection.
- Nutrition values come from a public table that anyone can audit.

## Mobile deployment

The detector is exported to **TFLite INT8** via Ultralytics' built-in
exporter. Portion estimation and nutrition lookup are pure-Python /
table-driven and trivially run on-device (no extra ML weights). The whole
on-device payload is ~3-4 MB for YOLOv8n + ~10 KB for the CSV.

## Vocabulary alignment

YOLO class names *must* match `tkpi_lookup.csv` `name_id`. This is enforced
implicitly because both are loaded from the same `configs/data.yaml`. To add
a new ingredient:

1. Add a row to `data/nutrition/tkpi_lookup.csv` with USDA / TKPI values.
2. Add the same name to `configs/data.yaml` under `names:` with a new index.
3. Re-run `scripts/prepare_yolo_dataset.py` and retrain.
