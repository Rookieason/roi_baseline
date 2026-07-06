# ROI Heatmap Ablation Results Summary

Source: `logs/run_20260705_115927.log`

## How to read this

- `6dof`: baseline model using pose history only.
- `heatmap_6dof`: same pose history plus compact RF heatmap features.
- `horizon_ms`: how far into the future the model predicts the ROI.
- `Top-k accuracy`: correct if the true ROI is within the model's top k predicted cells. Higher is better.
- `ROI distance error mean`: distance between predicted top-1 ROI center and true ROI center. Lower is better.
- `latency_ms_mean`: prediction time per sample. Lower is better.
- `peak_memory_mb`: evaluation peak memory. Lower is better.

## Main Takeaway

The RF heatmap features help substantially when the system can consider several candidate ROIs:

- Top-3 improves by about 9.6 to 10.5 percentage points.
- Top-5 improves by about 12.5 to 13.0 percentage points.
- Top-10 improves by about 15.1 to 15.3 percentage points.
- Mean ROI distance error improves by about 3.6 to 4.3 units.

The tradeoff is that exact Top-1 accuracy drops by about 0.8 to 1.4 percentage points, and memory rises from about 1.2 MB to about 42.7 MB. Latency rises only slightly, from about 0.12 ms to about 0.14-0.15 ms per prediction.

## Comparison Table

| Horizon | Model | Top-1 | Top-3 | Top-5 | Top-10 | Mean ROI Error | Mean Latency | Peak Memory | Test Samples |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 ms | 6dof | 14.36% | 31.11% | 39.33% | 57.01% | 46.91 | 0.124 ms | 1.25 MB | 17715 |
| 50 ms | heatmap_6dof | 12.92% | 40.69% | 52.28% | 72.32% | 43.13 | 0.144 ms | 42.75 MB | 15688 |
| 100 ms | 6dof | 14.42% | 31.24% | 39.69% | 57.29% | 46.80 | 0.123 ms | 1.21 MB | 17591 |
| 100 ms | heatmap_6dof | 13.48% | 41.04% | 52.45% | 72.59% | 42.47 | 0.154 ms | 42.73 MB | 15679 |
| 150 ms | 6dof | 14.35% | 31.56% | 39.84% | 57.31% | 46.56 | 0.127 ms | 1.20 MB | 17343 |
| 150 ms | heatmap_6dof | 13.29% | 41.32% | 52.60% | 72.39% | 42.73 | 0.148 ms | 42.73 MB | 15659 |
| 200 ms | 6dof | 14.23% | 31.55% | 40.82% | 57.72% | 46.60 | 0.128 ms | 1.17 MB | 17219 |
| 200 ms | heatmap_6dof | 13.43% | 42.01% | 53.27% | 72.96% | 42.96 | 0.148 ms | 42.75 MB | 15649 |

## Notes

- The heatmap model has fewer test samples because evaluation requires matched heatmap data.
- Median ROI distance error is identical in this run, so the mean distance error is more informative here.
- Training losses decrease for both models across all horizons, so the run appears to have completed normally.
- Only the wrapper log was copied back here. The formal `comparison_report.json` and `models/*.npz` files are not present in this local output tree; they were likely written under the dataset root on the machine where the experiment ran.
