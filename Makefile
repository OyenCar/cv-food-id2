.PHONY: help setup test lint smoke db-check demo notebook clean

help:
	@echo "cvfoodid -- common tasks"
	@echo ""
	@echo "  make setup       create venv, install editable + dev/demo extras"
	@echo "  make test        run pytest"
	@echo "  make lint        run ruff"
	@echo "  make smoke       run end-to-end smoke test (synthetic plate)"
	@echo "  make db-check    validate the bundled TKPI nutrition CSV"
	@echo "  make demo        launch Gradio demo (needs trained weights)"
	@echo "  make clean       remove caches"

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev,demo]"

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check src tests scripts app

smoke:
	. .venv/bin/activate && python scripts/smoke_test.py

db-check:
	. .venv/bin/activate && python scripts/build_nutrition_db.py --csv data/nutrition/tkpi_lookup.csv --check-only

demo:
	. .venv/bin/activate && python app/gradio_demo.py --weights $${WEIGHTS:-runs/detect/cvfoodid-yolo/weights/best.pt}

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/cvfoodid.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
