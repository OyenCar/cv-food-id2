# Improving portion accuracy

The default `MassEstimator` uses a simple geometric model
(`area * pile_height * density`). It is intentionally transparent and
debuggable -- but it is also approximate. This doc lists concrete ways to
push portion accuracy up.

## Tier 0 -- defaults (current)

* Pile heights from `DEFAULT_HEIGHTS_MM` and `INGREDIENT_HEIGHT_OVERRIDES_MM`.
* Density from `tkpi_lookup.csv`.
* Reference object preferred, plate fallback otherwise.

Expected error: ~ +- 30 % per ingredient on typical phone photos.

## Tier 1 -- better reference detection

Train the YOLO model to recognize **at least one reference class**. The
cheapest is a 1000-rupiah coin (24.5 mm diameter, common, copper-yellow
contrast pops out on most plates). Add ~50-100 cropped coin images to your
dataset under `coin_idr_1000` and retrain.

## Tier 2 -- depth from a single image

Use a pretrained monocular-depth model (DPT-Hybrid, MiDaS, Depth Anything v2)
to recover a per-pixel relative depth map. Combined with the reference
scale, this gives an actual height per detection rather than a category
default.

```python
# Sketch
depth = midas.estimate(image)
mask = detection_mask(image, det)
relative_height = (depth[mask] - depth[mask].min()).mean()
absolute_height_mm = relative_height * scale_factor  # calibrate against reference
```

Expected improvement: ~ +- 15 % per ingredient.

## Tier 3 -- learned regression head

Replace the geometric model entirely with a regression network conditioned
on (image crop, ingredient class, scale). Train on Nutrition5k for the
backbone and fine-tune on Indonesian dishes with self-collected mass labels
(use a kitchen scale).

Expected improvement: < 10 % per ingredient (matches the Nutrition5k paper).

## Tier 4 -- RGB-D / LiDAR

iPhone Pro / Pro Max devices have LiDAR; modern Android flagships have ToF
sensors. With true depth, mass estimation becomes a 3D-mesh + density
integration problem with sub-5 % error.

## Picking a tier

| Tier | Effort | Accuracy | Mobile-friendly? |
|---|---|---|---|
| 0 | Done | +- 30 % | Yes |
| 1 | A weekend | +- 25 % | Yes |
| 2 | Two weeks | +- 15 % | DPT-Hybrid (~80 MB) is borderline; quantize to INT8 |
| 3 | A month | +- 10 % | Add ~5 MB regression head |
| 4 | Hardware-dependent | +- 5 % | iPhone Pro / Android ToF only |

Most MVPs ship at Tier 0-1 and graduate to Tier 2-3 once they have user
data.
