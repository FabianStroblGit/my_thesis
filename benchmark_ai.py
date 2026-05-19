"""
benchmark_ai.py — Run Active Inference with and without thigmotaxis for N runs each.

Workflow:
  1. Snapshot existing data files.
  2. Run a fresh INIT (exploration) phase: from_data=false, nr_steps=nr_steps_exploration=14000,
     doors_option=plane_doors. This populates GC/PC/cognitive-map from scratch.
  3. Snapshot the init result.
  4. For each thigmotaxis setting, run N navigation episodes (from_data=true,
     150k steps, plane_doors_individual), restoring the init snapshot each time.
  5. Restore the original snapshot.

Saves a trajectory screenshot after every run to benchmark_ai_plots/.

Usage:
    python benchmark_ai.py              # 5 runs per config (default)
    python benchmark_ai.py --runs 10    # 10 runs per config
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.json"
PLOT_DIR = ROOT / "benchmark_ai_plots"

GC_FILES = [
    Path("data/gc_model/w_vectors.npy"),
    Path("data/gc_model/h_vectors.npy"),
    Path("data/gc_model/gm_values.npy"),
    Path("data/gc_model/s_vectors_initialized.npy"),
]
PC_FILES = [
    Path("data/pc_model/gc_connections.npy"),
    Path("data/pc_model/env_coordinates.npy"),
]
MAP_FILES = [
    Path("data/cognitive_map/topology_cells.npy"),
    Path("data/cognitive_map/reward_cells.npy"),
    Path("data/cognitive_map/recency_cells.npy"),
    Path("data/cognitive_map/protected_connections.npy"),
]
DATA_FILES = GC_FILES + PC_FILES + MAP_FILES

INIT_STEPS = 14000

DEBUG_PREFIXES = ("[PATH]", "[PRUNE]", "[THIGMO]", "[A2C]", "[AI]", "[PC]",
                  "[MOTOR]", "------ Sub goal", "Choose goal", "No goal vector",
                  "------ Goal localization", "Goal vector:", "[METRIC]")


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def run_with_config(label: str, cfg_override: dict, plot_path: str = None,
                    quiet: bool = False, live: bool = False) -> dict:
    """Run main.py with overridden config and return parsed metrics dict.

    When ``live`` is True, the subprocess keeps the interactive matplotlib
    backend so the LiveCognitiveMapPlot window appears during the run.
    """
    cfg = load_config()

    def deep_merge(base, override):
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                deep_merge(base[k], v)
            else:
                base[k] = v

    deep_merge(cfg, cfg_override)

    # Per-script suffix so parallel benchmark invocations don't race over
    # the same backup file. PID is included for safety against accidental
    # double-invocation of the same script.
    tmp_cfg = CONFIG_PATH.with_suffix(f".benchmark_ai_{os.getpid()}_tmp.json")
    backup = CONFIG_PATH.with_suffix(f".benchmark_ai_{os.getpid()}_backup.json")

    metrics = {}
    # Copy (not rename) the original so config.json stays on disk for the
    # whole subprocess lifetime — if this script is hard-killed there's no
    # window in which the user's config could disappear.
    shutil.copy2(CONFIG_PATH, backup)
    try:
        with tmp_cfg.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        tmp_cfg.replace(CONFIG_PATH)

        if not quiet:
            print(f"\n  --- {label} ---")

        env = os.environ.copy()
        if plot_path:
            env["BENCHMARK_PLOT_PATH"] = str(plot_path)
        # Headless matplotlib by default for benchmarks; interactive when
        # --live was passed so the user can watch the cog-map update.
        env["HEADLESS_MPL"] = "0" if live else "1"
        # Skip the per-lookahead diagnostic PDF saves — these are heavy I/O
        # and only useful when interactively debugging the lookahead.
        env["BENCHMARK_NO_PLOTS"] = "1"

        proc = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=ROOT,
            env=env,
        )

        for line in proc.stdout:
            m = re.search(r"\[METRICS\] (.+)", line)
            if m:
                try:
                    metrics = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

            # Only show progress, goal, and profiling lines
            if not quiet and (line.startswith("Progress:")
                              or "[GOAL]" in line
                              or "[PROF]" in line):
                sys.stdout.write(f"    {line}")
                sys.stdout.flush()

        proc.wait()

    finally:
        # Defensive restore — if the backup is missing (e.g. a parallel
        # benchmark deleted it under an older shared filename), log and skip
        # rather than crash.
        if backup.exists():
            shutil.copy2(backup, CONFIG_PATH)
            backup.unlink(missing_ok=True)
        else:
            print(f"[benchmark] WARNING: backup {backup.name} missing at "
                  f"restore time; leaving config.json as-is", file=sys.stderr)
        tmp_cfg.unlink(missing_ok=True)

    if plot_path and Path(plot_path).exists():
        print(f"    [benchmark] Plot saved to {plot_path}")

    return metrics


def snapshot_data(snapshot_dir: Path):
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for rel in DATA_FILES:
        src = ROOT / rel
        dst = snapshot_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)


def restore_data(snapshot_dir: Path):
    for rel in DATA_FILES:
        src = snapshot_dir / rel
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)


def summarize(runs: list[dict], metric_key: str) -> str:
    """Return 'mean +/- std' string for a numeric metric."""
    vals = [r.get(metric_key) for r in runs if r.get(metric_key) is not None]
    if not vals:
        return "n/a"
    arr = np.array(vals, dtype=float)
    return f"{np.mean(arr):.1f} +/- {np.std(arr):.1f}"


def main():
    parser = argparse.ArgumentParser(description="Active Inference benchmark with multiple runs")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per configuration")
    parser.add_argument("--doors", type=str, default="1",
                        help="Comma-separated list of door indices to evaluate "
                             "(1=leftmost x=1.5, ..., 5=rightmost x=9.5). "
                             "Each index runs N evaluation episodes with that door "
                             "as the ONLY open passage in the dividing wall. "
                             "Default '1' preserves the original benchmark; pass "
                             "'1,2,3,4,5' for full coverage.")
    parser.add_argument("--thigmo", choices=["on", "off", "both"], default="both",
                        help="Which thigmotaxis settings to evaluate. "
                             "'on' = only thigmotaxis enabled, "
                             "'off' = only disabled, "
                             "'both' = run both (default).")
    parser.add_argument("--live", action="store_true",
                        help="Show the live cognitive-map matplotlib window "
                             "during each evaluation run. Slower than the "
                             "default headless mode but useful for demos.")
    parser.add_argument("--seed-base", type=int, default=None,
                        help="If set, the k-th run within each (door, thigmo) "
                             "cell is launched with simulation.seed = "
                             "seed_base + k (numbering restarts at 0 per cell). "
                             "Diversifies numpy/random RNG state across runs so "
                             "the AIF planner is evaluated under independent "
                             "stochasticity rather than relying solely on "
                             "PyBullet contact-resolver noise.")
    args = parser.parse_args()
    n_runs = args.runs
    live = args.live
    # Only the V2 planner remains; the legacy v1 greedy planner has been removed.
    explorer_type = "active_inference_v2"
    if args.thigmo == "on":
        thigmo_settings = [True]
    elif args.thigmo == "off":
        thigmo_settings = [False]
    else:
        thigmo_settings = [True, False]

    try:
        door_list = [int(d.strip()) for d in args.doors.split(",") if d.strip()]
    except ValueError:
        sys.exit(f"[benchmark] --doors expects integers, got: {args.doors!r}")
    for d in door_list:
        if d not in (1, 2, 3, 4, 5):
            sys.exit(f"[benchmark] --doors entries must be in 1..5, got {d}")
        urdf = ROOT / "environment" / "linear_sunburst_map" / f"plane_doors_only_{d}.urdf"
        if not urdf.exists():
            sys.exit(f"[benchmark] missing URDF for door {d}: {urdf}\n"
                     f"            run: python environment/linear_sunburst_map/generate_door_variants.py")

    # Unique run-id stamps every output of this benchmark invocation so
    # consecutive runs do not overwrite each other's plots or results JSON.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir_run = PLOT_DIR / run_id

    print("\n" + "="*70)
    print(f"  ACTIVE INFERENCE BENCHMARK  —  init + 2 configs x {n_runs} runs  (150k step limit)")
    print(f"  Run id: {run_id}")
    print("="*70)

    plot_dir_run.mkdir(parents=True, exist_ok=True)

    # ── Step 1: snapshot original data ──
    original_snapshot = ROOT / ".benchmark_ai_original"
    snapshot_data(original_snapshot)
    print("[benchmark] Original data snapshot saved")

    # ── Step 2: run init (exploration) phase ──
    print(f"\n{'='*60}")
    print(f"  INIT  [exploration phase — from_data=false, plane_doors]")
    print(f"{'='*60}")

    init_plot = plot_dir_run / "00_init_exploration.png"
    run_with_config(
        "INIT",
        cfg_override={
            "simulation": {
                "nr_steps": INIT_STEPS,
                "nr_steps_exploration": INIT_STEPS,
            },
            "grid_cell_network": {"from_data": False},
            "environment": {"doors_option": "plane_doors"},
            # Match the A2C / Random benchmarks: skip the leftward sweep of
            # the lower corridor so the three leftmost PCs ([1.5,4.5],
            # [2.5,4.5], [3.5,4.5]) are never created during exploration.
            "exploration": {"skip_left_lower_init": True},
        },
        plot_path=str(init_plot),
        quiet=True,
        live=live,
    )
    print("[benchmark] Init (exploration) phase complete")

    # ── Step 3: snapshot init result ──
    init_snapshot = ROOT / ".benchmark_ai_init"
    snapshot_data(init_snapshot)
    print("[benchmark] Init data snapshot saved")

    # ── Step 4: run each config ──
    all_results = {}

    for door_idx in door_list:
        doors_option = f"plane_doors_only_{door_idx}"
        for thigmo in thigmo_settings:
            thigmo_str = "ON" if thigmo else "OFF"
            config_label = f"active_inference / door={door_idx} / thigmotaxis={thigmo_str}"
            print(f"\n{'='*60}")
            print(f"  {config_label}  ({n_runs} runs)")
            print(f"{'='*60}")

            runs = []
            for run_idx in range(n_runs):
                restore_data(init_snapshot)

                plot_file = plot_dir_run / (
                    f"ai_door{door_idx}_thigmo{thigmo_str}_run{run_idx+1:02d}.png"
                )

                sim_override = {
                    "nr_steps": 150000,
                    "nr_steps_exploration": 0,
                }
                if args.seed_base is not None:
                    sim_override["seed"] = int(args.seed_base) + run_idx

                metrics = run_with_config(
                    f"Run {run_idx+1}/{n_runs}  [{config_label}]"
                    + (f"  seed={sim_override['seed']}" if "seed" in sim_override else ""),
                    cfg_override={
                        "simulation": sim_override,
                        "grid_cell_network": {"from_data": True},
                        "environment": {"doors_option": doors_option},
                        "exploration": {"type": explorer_type},
                        "thigmotaxis": {"enabled": thigmo},
                        # Enable the live cog-map window when --live was passed.
                        "plotting": {"live_plot": live},
                    },
                    plot_path=str(plot_file),
                    live=live,
                )
                runs.append(metrics)

                reached = metrics.get("goal_reached", False)
                nav = metrics.get("nav_steps")
                print(f"    -> {'goal at nav step ' + str(nav) if reached else 'timeout'}")

            all_results[config_label] = runs

    # ── Step 5: restore original data ──
    restore_data(original_snapshot)
    print("\n[benchmark] Original data restored")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  ACTIVE INFERENCE BENCHMARK SUMMARY")
    print(f"{'='*70}")

    for config_label, runs in all_results.items():
        n_success = sum(1 for r in runs if r.get("goal_reached", False))
        nav_steps_vals = [r["nav_steps"] for r in runs if r.get("nav_steps") is not None]

        print(f"\n  {config_label}")
        print(f"  {'-'*50}")
        print(f"    Success rate:       {n_success}/{n_runs} ({n_success/n_runs*100:.0f}%)")

        if nav_steps_vals:
            arr = np.array(nav_steps_vals, dtype=float)
            print(f"    Nav steps (success): {np.mean(arr):.0f} +/- {np.std(arr):.0f}  "
                  f"(min={np.min(arr):.0f}, max={np.max(arr):.0f})")
        else:
            print(f"    Nav steps (success): n/a (no successful runs)")

        print(f"    Wall-hug fraction:  {summarize(runs, 'wall_hug_fraction')}")
        print(f"    Unique PCs visited: {summarize(runs, 'unique_pcs_visited')}")
        print(f"    First door step:    {summarize(runs, 'first_door_step')}")
        print(f"    Door crossings:     {summarize(runs, 'num_door_crossings')}")

    print(f"\n  Plots saved to {plot_dir_run}/")
    print(f"{'='*70}\n")

    # Save the aggregate summary (mean +/- std across runs) instead of
    # the raw per-run metrics. The thesis only ever consumes the
    # aggregate; the per-run JSON ballooned with every benchmark.
    from benchmark_summary import build_aggregate
    out_path = ROOT / f"benchmark_ai_results_{run_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(build_aggregate(all_results, n_runs), f, indent=2)
    print(f"[benchmark] Aggregate summary saved to {out_path}")


if __name__ == "__main__":
    main()
