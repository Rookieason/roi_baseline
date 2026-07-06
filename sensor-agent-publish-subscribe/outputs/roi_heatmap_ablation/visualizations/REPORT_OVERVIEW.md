# ROI Heatmap Ablation Report

## 1. Experiment goal

This experiment tests whether RF heatmap information helps predict a future region of interest (ROI). We compare two lightweight models:

- `6dof`: uses only headset pose history, meaning position and rotation over the previous 250 ms.
- `heatmap_6dof`: uses the same pose history plus compact features extracted from the RF heatmap.

The prediction task is not to estimate an exact coordinate directly. Instead, the physical 3D space is divided into ROI cells, and the model predicts which cell the future position will fall into.

## 2. ROI definition

The ROI grid size is `8` on each axis, so the space is divided into:

`8 x 8 x 8 = 512` ROI cells.

The coordinate bounds used in this run are:

- Min: x=-80.50, y=-115.06, z=71.87
- Max: x=91.29, y=182.81, z=186.22

Approximate cell size:

- X: 21.47
- Y: 37.23
- Z: 14.29

## 3. Dataset used

The split contains 825 sessions: 577 train, 123 validation, and 125 test.

The result report also records the number of aligned training and testing samples for each horizon. The sample count changes slightly by horizon because a future target must exist at 50, 100, 150, or 200 ms after the input timestamp. The heatmap model has fewer samples because it requires a matched heatmap in addition to pose data.

## 4. Prediction horizons

The experiment predicts four future time offsets:

- 50 ms
- 100 ms
- 150 ms
- 200 ms

A 50 ms horizon means: given current/recent information, predict the ROI 50 ms in the future. A 200 ms horizon is harder because it looks farther ahead.

## 5. What each metric means

- `Top-1 accuracy`: the model's first predicted ROI cell is exactly correct. Higher is better.
- `Top-3 / Top-5 / Top-10 accuracy`: the true ROI is within the first 3, 5, or 10 predicted cells. Higher is better. This is useful when the system can prefetch or prepare several candidate ROIs.
- `Mean ROI distance error`: distance between the center of the top-1 predicted ROI and the center of the true ROI. Lower is better.
- `Median ROI distance error`: the middle error value. It is less affected by extreme cases than the mean.
- `Mean latency`: average model prediction time per sample. Lower is better.
- `P95 latency`: 95% of predictions are faster than this value. Lower is better.
- `Peak memory`: peak memory observed during evaluation. Lower is better.
- `Training loss`: optimization objective during training. A decreasing loss indicates the model is learning the training data.

## 6. Main result

Adding heatmap features does not improve exact Top-1 accuracy in this run. On average, Top-1 changes by -1.06 percentage points.

However, heatmap features strongly improve candidate-set accuracy:

- Average Top-3 improvement: 9.90 percentage points.
- Average Top-5 improvement: 12.73 percentage points.
- Average Top-10 improvement: 15.23 percentage points.

Mean ROI distance error also improves by 3.89 units on average, meaning the heatmap model's first prediction is spatially closer even when it is not exactly the correct cell.

The tradeoff is resource cost. The heatmap model uses about 42.7 MB during evaluation, while the pose-only model uses about 1.2 MB. Prediction latency is still very small for both models, around 0.12-0.15 ms per sample.

## 7. How to present this in a report

A concise conclusion is:

> The RF heatmap features did not improve exact single-cell prediction, but they made the correct ROI much more likely to appear among the top candidate regions. This suggests heatmap features are useful for systems that can prefetch, rank, or inspect multiple candidate ROIs rather than relying only on the first predicted cell.

## 8. Figures

- ROI grid: `figures/roi_grid.svg`
- Dataset split: `figures/dataset_split.svg`
- Top-k accuracy: `figures/top1_accuracy.svg`, `figures/top3_accuracy.svg`, `figures/top5_accuracy.svg`, `figures/top10_accuracy.svg`
- Heatmap improvement: `figures/accuracy_delta.svg`
- ROI distance error: `figures/roi_error.svg`
- Latency: `figures/latency.svg`
- Memory: `figures/memory.svg`
- Training loss: `figures/training_loss.svg`

## 9. Full result table

| Horizon | Model | Train samples | Test samples | Top-1 | Top-3 | Top-5 | Top-10 | Mean ROI error | Mean latency | Peak memory |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 ms | 6DoF only | 82213 | 17715 | 14.36% | 31.11% | 39.33% | 57.01% | 46.91 | 0.124 ms | 1.25 MB |
| 50 ms | Heatmap + 6DoF | 72534 | 15688 | 12.92% | 40.69% | 52.28% | 72.32% | 43.13 | 0.144 ms | 42.75 MB |
| 100 ms | 6DoF only | 81638 | 17591 | 14.42% | 31.24% | 39.69% | 57.29% | 46.80 | 0.123 ms | 1.21 MB |
| 100 ms | Heatmap + 6DoF | 72497 | 15679 | 13.48% | 41.04% | 52.45% | 72.59% | 42.47 | 0.154 ms | 42.73 MB |
| 150 ms | 6DoF only | 80489 | 17343 | 14.35% | 31.56% | 39.84% | 57.31% | 46.56 | 0.127 ms | 1.20 MB |
| 150 ms | Heatmap + 6DoF | 72423 | 15659 | 13.29% | 41.32% | 52.60% | 72.39% | 42.73 | 0.148 ms | 42.73 MB |
| 200 ms | 6DoF only | 79913 | 17219 | 14.23% | 31.55% | 40.82% | 57.72% | 46.60 | 0.128 ms | 1.17 MB |
| 200 ms | Heatmap + 6DoF | 72386 | 15649 | 13.43% | 42.01% | 53.27% | 72.96% | 42.96 | 0.148 ms | 42.75 MB |
