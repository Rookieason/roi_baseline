from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from .data.discover_sessions import SessionInfo
from .data.roi_label import position_to_roi, roi_center
from .data.timestamp_alignment import iter_aligned_samples
from .models.baseline_6dof import SoftmaxClassifier
from .train import sample_features


def evaluate_one(
    sessions: list[SessionInfo],
    horizon_ms: int,
    model_kind: str,
    config: dict,
    model_path: Path,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> dict:
    model, standardizer = SoftmaxClassifier.load(model_path)
    top_ks = sorted(set(int(k) for k in config["evaluation"]["top_k"]))
    max_k = max(top_ks)
    grid_size = int(config["roi"]["grid_size"])
    correct = {k: 0 for k in top_ks}
    distances: list[float] = []
    latencies: list[float] = []
    total = 0

    tracemalloc.start()
    for session in sessions:
        for sample in iter_aligned_samples(
            session,
            horizon_ms=horizon_ms,
            pose_history_ms=config["pose_history_ms"],
            pose_history_size=config["pose_history_size"],
            tolerance_ms=config["alignment_tolerance_ms"],
            sample_stride=config.get("sample_stride", 1),
            require_heatmap=(model_kind != "6dof"),
        ):
            y = position_to_roi(sample.target_position, bounds_min, bounds_max, grid_size)
            x = sample_features(sample, model_kind, config)[None, :]
            x = standardizer.transform(x)
            start = time.perf_counter()
            pred_top = model.predict_topk(x, max_k)[0]
            latencies.append((time.perf_counter() - start) * 1000.0)
            for k in top_ks:
                correct[k] += int(y in pred_top[:k])
            distances.append(float(np.linalg.norm(roi_center(int(pred_top[0]), bounds_min, bounds_max, grid_size) - roi_center(y, bounds_min, bounds_max, grid_size))))
            total += 1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "samples": total,
        "top_k_accuracy": {str(k): (correct[k] / total if total else 0.0) for k in top_ks},
        "roi_distance_error_mean": float(np.mean(distances)) if distances else None,
        "roi_distance_error_median": float(np.median(distances)) if distances else None,
        "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
        "peak_memory_mb": peak / (1024 * 1024),
    }


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

