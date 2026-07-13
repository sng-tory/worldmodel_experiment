from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ACTION_STATS_FILENAME = "so100_action_statistics.json"


def default_challenge_action_stats_path(challenge_root: str | Path) -> Path:
    return Path(challenge_root) / DEFAULT_ACTION_STATS_FILENAME


def compute_challenge_action_statistics(challenge_root: str | Path) -> dict[str, Any]:
    action_dir = Path(challenge_root) / "actions"
    action_paths = sorted(action_dir.glob("*.npy"))
    if not action_paths:
        raise ValueError(f"No action npy files found under {action_dir}.")

    count = 0
    total = None
    total_sq = None
    action_dim = None
    for action_path in action_paths:
        actions = np.load(action_path).astype(np.float64)
        if actions.ndim != 2:
            raise ValueError(f"{action_path}: expected action shape (T, A), got {actions.shape}.")
        if action_dim is None:
            action_dim = actions.shape[-1]
            total = np.zeros(action_dim, dtype=np.float64)
            total_sq = np.zeros(action_dim, dtype=np.float64)
        elif actions.shape[-1] != action_dim:
            raise ValueError(f"{action_path}: expected action dim {action_dim}, got {actions.shape[-1]}.")

        count += actions.shape[0]
        total += actions.sum(axis=0)
        total_sq += np.square(actions).sum(axis=0)

    if count == 0 or total is None or total_sq is None:
        raise ValueError(f"Cannot compute action statistics from empty files under {action_dir}.")

    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    return {
        "count": int(count),
        "num_files": len(action_paths),
        "action_dim": int(action_dim),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


def load_or_compute_challenge_action_statistics(
    challenge_root: str | Path,
    action_stats_path: str | Path | None = None,
    *,
    write_if_missing: bool = True,
) -> dict[str, Any]:
    stats_path = Path(action_stats_path) if action_stats_path else default_challenge_action_stats_path(challenge_root)
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    stats = compute_challenge_action_statistics(challenge_root)
    if write_if_missing:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"[action stats] computed eval action statistics -> {stats_path}", flush=True)
    return stats
