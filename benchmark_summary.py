"""
benchmark_summary.py — shared helper for the benchmark scripts.

Builds the per-configuration aggregate summary (mean ± std across runs,
success rate, nav-step stats) that the benchmark drivers print to stdout
*and* now save to JSON. Earlier the JSON output was the raw per-run
metrics list; the thesis only ever consumes the aggregate, so the JSON
now mirrors the printed summary for direct downstream use.
"""

from __future__ import annotations

import numpy as np


def _stat(vals: list, key: str = None) -> dict:
    """Mean/std/min/max for a list. Missing list → None entries everywhere."""
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

    Input shape:
        all_results = {
            "active_inference / door=1 / thigmotaxis=ON": [run1_metrics, run2_metrics, ...],
            ...
        }
        where each runN_metrics is the dict main.py writes (goal_reached,
        nav_steps, wall_hug_fraction, unique_pcs_visited, first_door_step,
        num_door_crossings, ...).

    Output shape (one entry per config, mirrors the printed summary):
        {
            "active_inference / door=1 / thigmotaxis=ON": {
                "n_runs": 1,
                "n_success": 1,
                "success_rate": 1.0,
                "nav_steps_success": {"mean": ..., "std": ..., "min": ..., "max": ..., "n": ...},
                "wall_hug_fraction": {"mean": ..., "std": ..., ...},
                "unique_pcs_visited": {...},
                "first_door_step": {...},
                "num_door_crossings": {...},
            },
            ...
        }

    ``nav_steps_success`` only counts runs where the agent actually reached
    the goal (matching the printed "Nav steps (success)" line). All other
    metrics are over every run.
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
