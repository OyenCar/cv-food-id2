#!/usr/bin/env bash
# Bootstrap a fresh checkout: create venv, install editable + dev + demo extras,
# run smoke + tests + lint to verify everything works.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev,demo]"

echo
echo "==> Validating nutrition database"
python scripts/build_nutrition_db.py --csv data/nutrition/tkpi_lookup.csv --check-only

echo
echo "==> Running smoke test"
python scripts/smoke_test.py

echo
echo "==> Running unit tests"
pytest -q

echo
echo "==> Running ruff"
ruff check src tests scripts app

echo
echo "Setup complete. Next steps:"
echo "  - Train on Colab: notebooks/01_train_colab.ipynb"
echo "  - Download datasets: python scripts/download_datasets.py --help"
echo "  - Launch Gradio demo (after training): make demo"
