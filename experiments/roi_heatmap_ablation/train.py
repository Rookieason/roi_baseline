from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .data.discover_sessions import SessionInfo
from .data.roi_label import fit_bounds, position_to_roi
from .data.timestamp_alignment import iter_aligned_samples, iter_session_positions
from .features.heatmap_features import heatmap_compact_features
from .features.pose_features import pose_history_features
from .models.baseline_6dof import SoftmaxClassifier, StreamingStandardizer


def fit_roi_bounds(sessions: Iterable[SessionInfo], padding_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for session in sessions:
        positions = np.asarray(list(iter_session_positions(session)), dtype=np.float64)
        if positions.size:
            bmin, bmax = fit_bounds(positions, padding_ratio)
            mins.append(bmin)
            maxs.append(bmax)
    if not mins:
        raise RuntimeError("No pose positions found for ROI bounds.")
    return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)


def sample_features(sample, model_kind: str, config: dict) -> np.ndarray:
    pose = pose_history_features(
        sample.pose_history_timestamps_ms,
        sample.pose_history_positions,
        sample.pose_history_rotations_deg,
    )
    if model_kind == "6dof":
        return pose
    heat_cfg = config["heatmap"]
    heat = heatmap_compact_features(
        sample.heatmap,
        configured_shape=heat_cfg.get("shape"),
        top_k=heat_cfg.get("top_k", 5),
        pooled_shape=heat_cfg.get("pooled_shape", [8, 8]),
    )
    return np.concatenate([pose, heat]).astype(np.float32)


def iter_xy(
    sessions: Iterable[SessionInfo],
    horizon_ms: int,
    model_kind: str,
    config: dict,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> Iterator[tuple[np.ndarray, int]]:
    roi_cfg = config["roi"]
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
            y = position_to_roi(sample.target_position, bounds_min, bounds_max, int(roi_cfg["grid_size"]))
            yield sample_features(sample, model_kind, config), y


def _batches(iterator: Iterator[tuple[np.ndarray, int]], batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for x, y in iterator:
        xs.append(x)
        ys.append(y)
        if len(xs) == batch_size:
            yield np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)
            xs.clear()
            ys.clear()
    if xs:
        yield np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def train_one(
    train_sessions: list[SessionInfo],
    horizon_ms: int,
    model_kind: str,
    config: dict,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    output_path: Path,
) -> dict:
    train_cfg = config["training"]
    batch_size = int(train_cfg["batch_size"])
    grid_size = int(config["roi"]["grid_size"])
    num_classes = grid_size ** 3
    standardizer = StreamingStandardizer()

    first_dim = None
    sample_count = 0
    for x, _ in iter_xy(train_sessions, horizon_ms, model_kind, config, bounds_min, bounds_max):
        standardizer.partial_fit(x)
        first_dim = x.size
        sample_count += 1
    if first_dim is None:
        raise RuntimeError(f"No training samples for {model_kind} horizon {horizon_ms} ms.")

    model = SoftmaxClassifier(first_dim, num_classes, seed=int(config.get("seed", 7)))
    losses: list[float] = []
    for _ in range(int(train_cfg["epochs"])):
        epoch_losses = []
        stream = iter_xy(train_sessions, horizon_ms, model_kind, config, bounds_min, bounds_max)
        for x, y in _batches(stream, batch_size):
            x = standardizer.transform(x)
            epoch_losses.append(model.train_batch(x, y, float(train_cfg["learning_rate"]), float(train_cfg["l2"])))
        if epoch_losses:
            losses.append(float(np.mean(epoch_losses)))

    model.save(output_path, standardizer)
    return {"samples": sample_count, "losses": losses, "model_path": str(output_path)}


def save_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

