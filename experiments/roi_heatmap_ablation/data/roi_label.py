from __future__ import annotations

import numpy as np


def fit_bounds(positions: np.ndarray, padding_ratio: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    mins = np.nanmin(positions, axis=0).astype(np.float64)
    maxs = np.nanmax(positions, axis=0).astype(np.float64)
    span = np.maximum(maxs - mins, 1e-6)
    pad = span * padding_ratio
    return mins - pad, maxs + pad


def position_to_roi(position: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray, grid_size: int) -> int:
    rel = (np.asarray(position, dtype=np.float64) - bounds_min) / np.maximum(bounds_max - bounds_min, 1e-9)
    ijk = np.floor(np.clip(rel, 0.0, 1.0 - 1e-12) * grid_size).astype(int)
    return int(ijk[0] * grid_size * grid_size + ijk[1] * grid_size + ijk[2])


def roi_to_ijk(index: int, grid_size: int) -> np.ndarray:
    i = index // (grid_size * grid_size)
    rem = index % (grid_size * grid_size)
    return np.array([i, rem // grid_size, rem % grid_size], dtype=np.int64)


def roi_center(index: int, bounds_min: np.ndarray, bounds_max: np.ndarray, grid_size: int) -> np.ndarray:
    ijk = roi_to_ijk(index, grid_size).astype(np.float64)
    cell = (bounds_max - bounds_min) / grid_size
    return bounds_min + (ijk + 0.5) * cell

