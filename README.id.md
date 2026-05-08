# cvfoodid

> **Pipeline computer vision hibrida untuk makanan Indonesia: deteksi bahan -> estimasi porsi -> hitung gizi.**
> Mobile-friendly (YOLOv8/v11 + TFLite). Label dwibahasa (ID / EN). Lisensi MIT.

[English](README.md) | Bahasa Indonesia

---

## Apa ini

Skeleton siap-pakai untuk merilis fitur "foto piring -> info gizi" khusus masakan Indonesia. Pipeline dipecah menjadi 3 tahap independen:

```
        Foto
          |
          v
+----------------------+
| 1. YOLO detector     |  -> bbox + nama bahan
+----------------------+
          |
          v
+----------------------+
| 2. Portion estimator |  -> massa (gram), pakai referensi
+----------------------+      objek (koin/KTP) atau diameter piring
          |
          v
+----------------------+
| 3. Nutrition lookup  |  -> kalori / protein / lemak / karbo / serat
+----------------------+      pakai tabel ala TKPI Kemenkes
```

Pipeline-nya dirakit di [`src/cvfoodid/pipeline.py`](src/cvfoodid/pipeline.py); setiap tahap bisa diganti / di-upgrade tanpa ganggu yang lain.

## Cepat dipakai

```bash
git clone <repo-ini>
cd cv-food-id

# (1) Install dependensi
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# (2) Smoke test (tanpa GPU, tanpa weight model)
python scripts/smoke_test.py

# (3) Unit tests
pytest -q

# (4) Lint
ruff check src tests scripts app
```

## Latih YOLOv8 untuk deteksi bahan

```bash
# 1. Download FoodSeg103 (Apache 2.0, boleh komersial) + dataset Indonesia.
python scripts/download_datasets.py foodseg103 padang-cuisine indonesian-traditional-foods

# 2. Konversi mask FoodSeg103 ke label bbox YOLO.
python scripts/prepare_yolo_dataset.py \
    --raw data/raw/foodseg103/FoodSeg103 \
    --out data/processed/yolo \
    --data-yaml configs/data.yaml

# 3. Training (cocok di Colab T4 / Kaggle P100).
python scripts/train_yolo.py \
    --data configs/data.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --name cvfoodid-yolo
```

Walkthrough lengkap di Colab: [`notebooks/01_train_colab.ipynb`](notebooks/01_train_colab.ipynb).

## Export ke mobile (TFLite, INT8)

```bash
python scripts/export_tflite.py \
    --weights runs/detect/cvfoodid-yolo/weights/best.pt \
    --imgsz 640 \
    --calib data/processed/yolo/images/val \
    --int8
```

`best_int8.tflite` yang dihasilkan kira-kira 3-4 MB di 640x640 dengan YOLOv8n. Tinggal masukin ke project Android (TFLite Java/Kotlin) atau iOS (konversi ke Core ML).

## Demo

```bash
python app/gradio_demo.py --weights runs/detect/cvfoodid-yolo/weights/best.pt
```

Buka <http://localhost:7860>, upload foto. Output: gambar dengan bbox + total kalori/makro dalam Bahasa Indonesia atau Inggris.

## Struktur folder

```
cv-food-id/
|-- configs/                  # data.yaml, training.yaml, model yamls
|-- data/
|   |-- nutrition/            # CSV gizi bahan ala TKPI
|   `-- raw/, processed/      # diisi oleh skrip download/prepare (gitignored)
|-- notebooks/                # walkthrough Colab
|-- scripts/                  # entry-point perintah
|-- src/cvfoodid/             # kode library
|-- app/gradio_demo.py
|-- tests/
|-- pyproject.toml
`-- README.md
```

## Catatan lisensi dataset

Lisensi dataset macam-macam. [`scripts/download_datasets.py`](scripts/download_datasets.py) sudah mencantumkannya per sumber.

| Dataset | Ukuran | Kegunaan utama | Lisensi | Boleh komersial? |
|---|---|---|---|---|
| **FoodSeg103 / 154** | ~9,5k gambar, 103-154 mask bahan | Tahap 1 deteksi bahan halus | Apache 2.0 | Ya |
| **FoodInsSeg** | 7,1k gambar, 119k instance mask | Tahap 1 instance segmentation | Academic only | Tidak |
| **Padang Cuisine (Kaggle)** | ~1k gambar, 9 kelas | Variasi makanan Indonesia | CC0 | Ya |
| **Indonesian Food (Mendeley)** | ~1k gambar, 10 kelas | Variasi makanan Indonesia | CC BY 4.0 | Ya (atribusi) |
| **BEEI 2024 Indonesian Food** | 24,4k gambar, 160 kelas | Dataset Indonesia terbesar | CC BY-SA | Ya (share-alike) |
| **In-TFK** | 1,6k + 1k gambar, 34 kelas | Foto studio kualitas tinggi | Academic | Cek dulu kalau komersial |
| **eriko-syah/indonesian-food** (HF) | 1,35k baris | Tabel kalori/protein/lemak/karbo per nama makanan ID | Tidak jelas | Anggap research |
| **Nutrition5k** (Google) | ~5k piring, RGB-D, massa per bahan | Tahap 2 porsi + Tahap 3 gizi | Research only | **Tidak** |
| **Recipe1M+** (MIT) | 13M gambar, 1M resep | Pretraining cross-modal | Research only | **Tidak** |
| **TKPI Kemenkes** | tabular | Tabel gizi bahan Indonesia | Publik | Ya |
| **USDA FoodData Central** | tabular | Tabel gizi internasional | Public domain | Ya |

Untuk MVP komersial: pakai **FoodSeg103 + BEEI 2024 + Padang + Mendeley** untuk training, dan **TKPI + USDA** untuk tabel gizi. Nutrition5k / Recipe1M+ hanya untuk benchmark research, jangan dishipping.

## Cara kerja estimasi porsi

Foto makanan single-image tidak punya skala absolut: mangkok nasi 12 cm dan piring nasi 24 cm bisa menghasilkan jumlah pixel yang sama. Kita atasi di [`src/cvfoodid/portion/reference.py`](src/cvfoodid/portion/reference.py) lewat 3 tingkat fallback:

1. **Objek referensi yang dikenali** dideteksi oleh YOLO yang sama: koin Indonesia (`coin_idr_500`, 27 mm), KTP (`ktp`, 85,6 mm), atau sendok standar. Confidence tertinggi (~0,95).
2. **Fitting elips piring** (default diameter 24 cm). Confidence menengah (~0,75).
3. **Heuristik ukuran gambar** asumsi jarak makan tipikal 30 cm. Confidence rendah (~0,30).

Massa lalu dihitung secara geometrik:

```
mass_g = projected_area_mm2 * effective_height_mm * density_g_per_ml / 1000
```

`effective_height_mm` adalah tinggi tumpukan empiris per kategori (nasi ~28 mm, kuah ~4 mm, daging ~15 mm). Untuk akurasi lebih tinggi, ganti dengan depth RGB-D (pola Nutrition5k) atau head regresi yang dilatih.

## Tabel gizi

[`data/nutrition/tkpi_lookup.csv`](data/nutrition/tkpi_lookup.csv) berisi **100 bahan umum Indonesia**, lengkap dengan:

- `kcal_per_100g`, `protein_g`, `fat_g`, `carb_g`, `fiber_g`, `water_g` (per 100 g)
- `density_g_per_ml` (dipakai estimator porsi)
- `category` (staple / protein / vegetable / aromatic / sauce / ...)
- `name_id`, `name_en` untuk render dwibahasa
- `source` (USDA / TKPI)

Validasi:

```bash
python scripts/build_nutrition_db.py --csv data/nutrition/tkpi_lookup.csv --check-only
```

## Pemakaian programatik

```python
from cvfoodid.nutrition import IngredientPortion, NutritionCalculator
from cvfoodid.portion import DetectionInput, MassEstimator, from_known_object

calc = NutritionCalculator()
estimator = MassEstimator()

ref = from_known_object((100, 100, 149, 149), "coin_idr_1000")  # mm/px
portions = []
for det in deteksi_anda:  # bisa dari model apa saja
    mass = estimator.estimate(
        DetectionInput(det.cls, det.bbox, det.conf), ref.mm_per_pixel
    )
    portions.append(IngredientPortion(det.cls, mass, det.conf))

result = calc.compute(portions)
print(result.as_dict())
```

## Roadmap

- [x] Tahap 1 (deteksi): trainer YOLOv8 + converter FoodSeg103 -> YOLO
- [x] Tahap 2 (porsi): estimator geometris dengan 3 tingkat scale recovery
- [x] Tahap 3 (gizi): tabel TKPI bundled + kalkulator
- [x] CLI + demo Gradio
- [x] Export TFLite INT8
- [ ] Sample app Android (TFLite + kamera)
- [ ] Sample app iOS (Core ML)
- [ ] Active learning dengan koreksi user
- [ ] Estimator porsi sadar-depth (MiDaS / DPT)
- [ ] Baseline VLM untuk caption (CLIP / VLM)

## Lisensi

MIT untuk kode di repo ini; lihat [LICENSE](LICENSE). Dataset & weight pretrained yang kamu download tunduk pada **lisensinya masing-masing** -- baca dulu sebelum dishipping.
