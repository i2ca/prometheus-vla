import json
import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def load_relative_stats(path: str | Path) -> dict:
    """Load relative_stats.json from a dataset directory."""
    path = Path(path)
    with open(path) as f:
        return json.load(f)


def load_stats(path: str | Path) -> dict:
    """Load stats.json from a dataset directory."""
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)
    # Convert lists to numpy arrays
    stats = {}
    for key, val in raw.items():
        stats[key] = {k: np.array(v, dtype=np.float32) for k, v in val.items()}
    return stats

def discover_task_dirs(data_path: str | Path) -> list[Path]:
    """Find all sub-directories that contain meta/info.json (lerobot datasets)."""
    data_path = Path(data_path)
    task_dirs = []
    for d in sorted(data_path.iterdir()):
        if d.is_dir() and ((d / "meta" / "info.json").exists() or (d / "meta" / "relative_stats.json").exists()):
            task_dirs.append(d)
    return task_dirs

def get_normalizer(stats: dict, key: str, norm_type: str = "minmax_q"):
    """Get normalization parameters for a given key.

    Args:
        stats: stats dict (from stats.json or relative_stats)
        key: feature key
        norm_type: "minmax_q" (q01/q99 → [-1,1]), "zscore" (mean/std), "minmax" (min/max → [-1,1])
    Returns:
        (offset, scale) such that normalized = (value - offset) / scale
    """
    if key not in stats:
        # Return identity normalization if key not found
        return np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)

    s = stats[key]
    if norm_type == "minmax_q":
        # Try global first, fallback to per-sample
        if "global_q01" in s:
            low = np.array(s["global_q01"], dtype=np.float32)
            high = np.array(s["global_q99"], dtype=np.float32)
        elif "q01" in s:
            low = np.array(s["q01"], dtype=np.float32)
            high = np.array(s["q99"], dtype=np.float32)
        elif "min" in s and "max" in s:
            print(f"Warning: using min/max for {key} normalization; consider recomputing stats with quantiles for better robustness.")
            low = np.array(s["min"], dtype=np.float32)
            high = np.array(s["max"], dtype=np.float32)
        else:
            print(f"Warning: no quantile stats for {key}, using identity normalization")
            return np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
        offset = (low + high) / 2
        scale = (high - low) / 2
        scale = np.where(scale < 1e-6, 1.0, scale)
        return offset, scale
    elif norm_type == "zscore":
        if "global_mean" in s:
            mean = np.array(s["global_mean"], dtype=np.float32)
            std = np.array(s["global_std"], dtype=np.float32)
        elif "mean" in s:
            mean = np.array(s["mean"], dtype=np.float32)
            std = np.array(s["std"], dtype=np.float32)
        else:
            return np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        return mean, std
    elif norm_type == "minmax":
        if "global_min" in s:
            low = np.array(s["global_min"], dtype=np.float32)
            high = np.array(s["global_max"], dtype=np.float32)
        elif "min" in s:
            low = np.array(s["min"], dtype=np.float32)
            high = np.array(s["max"], dtype=np.float32)
        else:
            return np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
        offset = (low + high) / 2
        scale = (high - low) / 2
        scale = np.where(scale < 1e-6, 1.0, scale)
        return offset, scale
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


def normalize(value: np.ndarray, offset: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Normalize value: (value - offset) / scale."""
    return (value - offset) / scale
