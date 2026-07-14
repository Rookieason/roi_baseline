# ROI heatmap ablation summary

Source log: `baseline\experiments\roi_heatmap_ablation\outputs\roi_heatmap_ablation\logs\run_20260712_150358.log`

## Takeaway

Heatmap + pose is better on overlap-oriented ROI metrics, especially IoU, F1, and recall. Exact full-mask match is lower, and evaluation uses fewer heatmap samples because samples without heatmaps are excluded.

## Mean test metrics

| model        | model_label    | iou    | f1     | precision | recall | per_cell_accuracy | exact_match_accuracy | latency_ms_mean | peak_memory_mb |
| ------------ | -------------- | ------ | ------ | --------- | ------ | ----------------- | -------------------- | --------------- | -------------- |
| 6dof         | Pose only      | 0.1913 | 0.3211 | 0.5817    | 0.2218 | 0.7467            | 0.1464               | 0.0915          | 1.0018         |
| heatmap_6dof | Heatmap + pose | 0.3512 | 0.5198 | 0.6892    | 0.4174 | 0.7878            | 0.1177               | 0.1151          | 42.3673        |

## Mean absolute change: heatmap + pose minus pose only

| metric               | delta_abs |
| -------------------- | --------- |
| exact_match_accuracy | -0.0287   |
| f1                   | 0.1987    |
| iou                  | 0.1600    |
| per_cell_accuracy    | 0.0411    |
| precision            | 0.1075    |
| recall               | 0.1956    |

## Mean added cost

| metric          | delta_abs |
| --------------- | --------- |
| latency_ms_mean | 0.0236    |
| peak_memory_mb  | 41.3655   |

## Best heatmap horizon by IoU

- Horizon: 50 ms
- IoU: 0.3585
- F1: 0.5278
- Recall: 0.4269
- Precision: 0.6912

## Caveat

The two models are not evaluated on exactly the same number of samples: the heatmap model requires heatmap availability, while pose only does not. For a strict apples-to-apples comparison, rerun pose-only evaluation on the heatmap-available subset too.
