"""benchmark_summary.py — shared aggregate-summary helper for the benchmark scripts."""

from __future__ import annotations

import numpy as np


def _stat(vals: list, key: str = None) -> dict:
    """Mean/std/min/max for a list. Empty list yields None entries."""
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def build_aggregate(all_results: dict, n_runs: int) -> dict:
    """Build the aggregate summary dict from per-run results.

    ``nav_steps_success`` only counts runs where the agent reached the
    goal; all other metrics are over every run.
    """
    summary: dict = {}
    for config_label, runs in all_results.items():
        n_success = sum(1 for r in runs if r.get("goal_reached", False))
        nav_success = [r["nav_steps"] for r in runs
                       if r.get("goal_reached", False) and r.get("nav_steps") is not None]

        def collect(key: str) -> list:
            return [r.get(key) for r in runs if r.get(key) is not None]

        summary[config_label] = {
            "n_runs": n_runs,
            "n_success": n_success,
            "success_rate": float(n_success / n_runs) if n_runs else 0.0,
            "nav_steps_success": _stat(nav_success),
            "wall_hug_fraction": _stat(collect("wall_hug_fraction")),
            "unique_pcs_visited": _stat(collect("unique_pcs_visited")),
            "first_door_step": _stat(collect("first_door_step")),
            "num_door_crossings": _stat(collect("num_door_crossings")),
        }
    return summary
