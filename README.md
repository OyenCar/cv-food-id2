# cvfoodid

> **Hybrid CV pipeline for Indonesian food: ingredient detection -> portion estimation -> nutrition lookup.**
> Mobile-friendly (YOLOv8/v11 + TFLite). Bilingual labels (ID / EN). MIT licensed.

[Bahasa Indonesia](README.id.md) | English

---

## What this is

A complete scaffold for shipping a "snap a photo of your plate, get the nutrition" feature for Indonesian dishes. The pipeline is split into three independently testable stages:

```
        Photo
          |
          v
+----------------------+
| 1. YOLO detector     |  -> ingredient bounding boxes & class names
+----------------------+
          |
          v
+----------------------+
| 2. Portion estimator |  -> mass in grams, using a reference object
+----------------------+      or a fitted plate ellipse for absolute scale
          |
          v
+----------------------+
| 3. Nutrition lookup  |  -> kcal / protein / fat / carb / fiber
+----------------------+      against a TKPI-style table
```

The stages are wired together in [`src/cvfoodid/pipeline.py`](src/cvfoodid/pipeline.py); each can be replaced or upgraded independently.

## Quickstart

```bash
git clone <this-repo>
cd cv-food-id

# (1) Install deps
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# (2) Run the pure-Python smoke test (no GPU, no model weights needed)
python scripts/smoke_test.py

# (3) Run the unit tests
pytest -q

# (4) Lint
ruff check src tests scripts app
```

## Train a YOLOv8 ingredient detector

```bash
# 1. Download FoodSeg103 (Apache 2.0, commercial OK) + Indonesian datasets.
python scripts/download_datasets.py foodseg103 padang-cuisine indonesian-traditional-foods

# 2. Convert FoodSeg103 masks to YOLO bbox labels.
python scripts/prepare_yolo_dataset.py \
    --raw data/raw/foodseg103/FoodSeg103 \
    --out data/processed/yolo \
    --data-yaml configs/data.yaml

# 3. Train (Colab T4 / Kaggle P100 friendly).
python scripts/train_yolo.py \
    --data configs/data.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --name cvfoodid-yolo
```

The end-to-end training-on-Colab walkthrough lives in [`notebooks/01_train_colab.ipynb`](notebooks/01_train_colab.ipynb).

## Export to mobile (TFLite, INT8)

```bash
python scripts/export_tflite.py \
    --weights runs/detect/cvfoodid-yolo/weights/best.pt \
    --imgsz 640 \
    --calib data/processed/yolo/images/val \
    --int8
```

The resulting `best_int8.tflite` is ~3-4 MB at 640x640 with YOLOv8n. Drop it into Android (TFLite Java/Kotlin bindings) or iOS (Core ML conversion) projects.

## Demo

```bash
python app/gradio_demo.py --weights runs/detect/cvfoodid-yolo/weights/best.pt
```

Open <http://localhost:7860> and upload a photo. The demo annotates the image and prints calorie / macro totals in Indonesian or English.

## Repository layout

```
cv-food-id/
|-- configs/                  # data.yaml, training.yaml, model yamls
|-- data/
|   |-- nutrition/            # TKPI-derived ingredient nutrition CSV
|   `-- raw/, processed/      # populated by download / prepare scripts (gitignored)
|-- notebooks/                # Colab-ready training & inference walkthroughs
|-- scripts/
|   |-- download_datasets.py
|   |-- prepare_yolo_dataset.py
|   |-- build_nutrition_db.py
|   |-- train_yolo.py
|   |-- export_tflite.py
|   `-- smoke_test.py
|-- src/cvfoodid/
|   |-- detection/            # YOLO wrapper
|   |-- portion/              # mass estimation + reference scaling
|   |-- nutrition/            # database + calculator
|   |-- pipeline.py           # end-to-end glue
|   |-- cli.py
|   `-- i18n.py
|-- app/gradio_demo.py
|-- tests/
|-- pyproject.toml
`-- README.md
```

## Datasets and licenses

We pull from a mix of public sources. Licenses vary; [`scripts/download_datasets.py`](scripts/download_datasets.py) lists them per source.

| Dataset | Size | Best for | License | Commercial OK? |
|---|---|---|---|---|
| **FoodSeg103 / 154** | ~9.5k images, 103-154 ingredient masks | Stage 1 fine-grained ingredient detection | Apache 2.0 | Yes |
| **FoodInsSeg** | 7.1k images, 119k instance masks | Stage 1 instance segmentation | Academic only | No |
| **Padang Cuisine (Kaggle)** | ~1k images, 9 classes | Indonesian dish coverage | CC0 | Yes |
| **Indonesian Food (Mendeley)** | ~1k images, 10 classes | Indonesian dish coverage | CC BY 4.0 | Yes (with attribution) |
| **BEEI 2024 Indonesian Food** | 24.4k images, 160 classes | Largest Indonesian dish set | CC BY-SA | Yes (share-alike) |
| **In-TFK** | 1.6k + 1k images, 34 classes | Studio-quality Indonesian food | Academic | Check before commercial use |
| **eriko-syah/indonesian-food** (HF) | 1.35k rows | Per-dish kcal/protein/fat/carb table | Unspecified | Treat as research |
| **Nutrition5k** (Google) | ~5k plates, RGB-D, per-ingredient mass | Stage 2 portion + Stage 3 nutrition | Research only | **No** |
| **Recipe1M+** (MIT) | 13M images, 1M recipes | Cross-modal pretraining | Research only | **No** |
| **TKPI Kemenkes** | tabular | Nutrition lookup ground truth | Public | Yes |
| **USDA FoodData Central** | tabular | Nutrition lookup fallback | Public domain | Yes |

For a commercial MVP, lean on **FoodSeg103 + BEEI 2024 + Padang + Mendeley** for training, and **TKPI + USDA** for the nutrition table. Only use Nutrition5k / Recipe1M+ for research benchmarks.

## How portion estimation works

Single-image food photos do not contain absolute scale: a 12 cm bowl of rice and a 24 cm plate of rice can produce identical pixel counts. We resolve this in [`src/cvfoodid/portion/reference.py`](src/cvfoodid/portion/reference.py) using a 3-tier fallback:

1. **Known reference object** detected by the same YOLO model: an Indonesian coin (`coin_idr_500`, 27 mm), a national ID card (`ktp`, 85.6 mm), or a standard spoon. Highest confidence (~0.95).
2. **Plate ellipse fit** (default plate diameter 24 cm). Medium confidence (~0.75).
3. **Image-size heuristic** assuming a typical 30 cm phone-eating distance. Low confidence (~0.30).

Mass is then estimated geometrically:

```
mass_g = projected_area_mm2 * effective_height_mm * density_g_per_ml / 1000
```

`effective_height_mm` is a per-category empirical pile height (rice ~28 mm, sauces ~4 mm, meat slabs ~15 mm). For higher accuracy, plug in RGB-D depth (Nutrition5k pattern) or a learned regression head.

## Nutrition lookup

[`data/nutrition/tkpi_lookup.csv`](data/nutrition/tkpi_lookup.csv) ships with **100 common Indonesian ingredients**, each annotated with:

- `kcal_per_100g`, `protein_g`, `fat_g`, `carb_g`, `fiber_g`, `water_g` (per 100 g)
- `density_g_per_ml` (used by the portion estimator)
- `category` (staple / protein / vegetable / aromatic / sauce / ...)
- `name_id`, `name_en` for bilingual rendering
- `source` (USDA / TKPI)

Validate with::

```bash
python scripts/build_nutrition_db.py --csv data/nutrition/tkpi_lookup.csv --check-only
```

## Programmatic use

```python
from cvfoodid.nutrition import IngredientPortion, NutritionCalculator
from cvfoodid.portion import DetectionInput, MassEstimator, from_known_object

calc = NutritionCalculator()
estimator = MassEstimator()

ref = from_known_object((100, 100, 149, 149), "coin_idr_1000")  # mm/px
portions = []
for det in your_detections:  # from any model, not just YOLO
    mass = estimator.estimate(
        DetectionInput(det.cls, det.bbox, det.conf), ref.mm_per_pixel
    )
    portions.append(IngredientPortion(det.cls, mass, det.conf))

result = calc.compute(portions)
print(result.as_dict())
```

## Roadmap

- [x] Stage 1 (detection): YOLOv8 trainer + FoodSeg103 -> YOLO converter
- [x] Stage 2 (portion): geometric estimator with 3-tier scale recovery
- [x] Stage 3 (nutrition): bundled TKPI table + calculator
- [x] CLI + Gradio demo
- [x] TFLite INT8 export
- [ ] Android sample app (TFLite + camera)
- [ ] iOS sample app (Core ML)
- [ ] Active learning loop with user corrections
- [ ] Depth-aware portion estimator (MiDaS / DPT)
- [ ] Vision-language captioning baseline (CLIP / VLM)

## License

MIT for the code in this repo; see [LICENSE](LICENSE). Datasets and pretrained weights you download are subject to **their own** licenses -- review them before shipping.
