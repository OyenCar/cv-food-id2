# Contributing

## Quick checks before opening a PR

```bash
. .venv/bin/activate
pytest -q
ruff check src tests scripts app
python scripts/build_nutrition_db.py --check-only
python scripts/smoke_test.py
```

All four should pass.

## Adding a new ingredient

1. Add a row to `data/nutrition/tkpi_lookup.csv`. Source the values from USDA
   (https://fdc.nal.usda.gov/) or TKPI; keep `density_g_per_ml` realistic.
2. Add the same `name_id` under `configs/data.yaml::names`.
3. If the ingredient is also in FoodSeg103, add the mapping to
   `FOODSEG103_TO_TKPI` in `scripts/prepare_yolo_dataset.py`.
4. Re-run `python scripts/build_nutrition_db.py --check-only` to verify.
5. Re-train the detector.

## Adding a new dataset

1. Add a `Source(...)` entry to `scripts/download_datasets.py::SOURCES`.
2. Implement (or reuse) a handler. The four built-ins are
   `http_zip`, `git`, `kaggle`, `manual`.
3. Update `data/README.md` with the new path layout.
4. Update the dataset license table in `README.md`.

## Coding conventions

* Python 3.10+, full type annotations.
* `from __future__ import annotations` at the top of every module.
* Pure-Python paths (calculator, estimator, database) must stay importable
  without `ultralytics` / `torch`.
* New public functions need a docstring with at least a one-line summary.

## Tests

* Unit tests live in `tests/`.
* Test names follow `test_<module>_<behavior>`.
* No network, no model downloads in tests; mock or skip if needed.
