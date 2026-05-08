# Training on Colab / Kaggle

This project is designed to train end-to-end on a free Colab T4. Below are
the practical knobs.

## Free Colab T4 recipe

| Setting | Value | Why |
|---|---|---|
| Model | `yolov8n.pt` | 3.2 M params, fits in T4 memory at imgsz 640 |
| imgsz | 640 | Sweet spot between accuracy and mobile latency |
| batch | 16 | Largest that fits with mixed precision on T4 |
| epochs | 100 | FoodSeg103 + Indonesian set converges around 80-100 |
| optimizer | AdamW (Ultralytics default) | Better than SGD on small datasets |
| augmentations | Mosaic 1.0, MixUp 0.1, HSV (S 0.7, V 0.4) | Robust to phone-camera lighting |

If you switch to L4 / A100 or Pro+, bump `batch` to 32 and try `yolov8s.pt`
(~11 M params; still TFLite-friendly).

## Kaggle P100 recipe

Kaggle P100 has 16 GB VRAM (same as T4). Same defaults work but disable
`workers=8` if you hit RAM limits with the bigger Indonesian merged dataset.

## Common gotchas

* **FoodSeg103 zip is password-protected** -- the password is
  `LARCdataset9947` (per the project's homepage). The download script prints a
  fallback `unzip` command when the auto-extract fails.
* **Class index mismatch** -- after editing `configs/data.yaml`, also rerun
  `scripts/prepare_yolo_dataset.py`. The mapping
  `FOODSEG103_TO_TKPI` inside that script is what bridges the two
  vocabularies.
* **Out-of-memory at validation time** -- pass `--imgsz 512` for val only:
  the bbox metrics are largely insensitive to input size.
* **TFLite export fails with `tensorflow not found`** -- run
  `pip install tensorflow` (the `mobile` extra) *before* `export_tflite.py`.
* **Different mAP between Colab and Kaggle** -- Ultralytics seeds RNGs but
  CUDA non-determinism is unavoidable. Expect ~ +- 0.005 mAP run-to-run.

## Resuming a crashed training

```bash
python scripts/train_yolo.py --resume \
    --data configs/data.yaml \
    --name cvfoodid-yolo
```

The Ultralytics resume logic picks up from the last `last.pt` checkpoint.
