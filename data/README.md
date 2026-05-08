# Data layout

This directory holds **only** the small bundled artefacts (`nutrition/tkpi_lookup.csv` and `nutrition/dish_to_ingredients.csv`). The large dataset zips are **not committed** -- they are downloaded by `scripts/download_datasets.py` and live under `data/raw/` (gitignored).

```
data/
|-- nutrition/
|   |-- tkpi_lookup.csv             <-- bundled (100 ingredients, per-100g nutrition)
|   `-- dish_to_ingredients.csv     <-- bundled (dish recipes for fallback)
|-- raw/                            <-- gitignored, populated by download_datasets.py
|   |-- foodseg103/
|   |-- padang-cuisine/
|   |-- indonesian-traditional-foods/
|   |-- ...
|-- processed/                      <-- gitignored, populated by prepare_yolo_dataset.py
|   `-- yolo/
|       |-- images/{train,val}/
|       `-- labels/{train,val}/
|-- interim/                        <-- gitignored, scratch space
`-- external/                       <-- gitignored, third-party bits
```

See [`scripts/download_datasets.py`](../scripts/download_datasets.py) for the exact list of supported sources, their licenses, and download handlers.
