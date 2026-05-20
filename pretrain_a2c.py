"""pretrain_a2c.py — Pre-train an A2C policy across N episodes and save the final checkpoint."""

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
DEFAULT_OUTPUT = ROOT / "data" / "a2c_pretrained.pt"

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
NAV_STEPS = 150000


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def run_with_config(label: str, cfg_override: dict, quiet: bool = True) -> dict:
    """Run main.py with overridden config and return parsed metrics dict."""
    cfg = load_config()

    def deep_merge(base, override):
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                deep_merge(base[k], v)
            else:
                base[k] = v

    deep_merge(cfg, cfg_override)

    tmp_cfg = CONFIG_PATH.with_suffix(".pretrain_tmp.json")
    backup = CONFIG_PATH.with_suffix(".pretrain_backup.json")

    metrics = {}
    shutil.copy2(CONFIG_PATH, backup)
    try:
        with tmp_cfg.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        tmp_cfg.replace(CONFIG_PATH)

        if not quiet:
            print(f"\n  --- {label} ---")

        proc = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=ROOT,
            env=os.environ.copy(),
        )

        for line in proc.stdout:
            m = re.search(r"\[METRICS\] (.+)", line)
            if m:
                try:
                    metrics = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
            if not quiet and (
                line.startswith("Progress:")
                or "[GOAL]" in line
                or "[A2C]" in line
            ):
                sys.stdout.write(f"    {line}")
                sys.stdout.flush()

        proc.wait()
    finally:
        shutil.copy2(backup, CONFIG_PATH)
        backup.unlink(missing_ok=True)
        tmp_cfg.unlink(missing_ok=True)

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


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train an A2C policy across multiple navigation episodes."
    )
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of training episodes (default: 10)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Where to write the final checkpoint "
                             f"(default: {DEFAULT_OUTPUT.relative_to(ROOT)})")
    parser.add_argument("--resume", action="store_true",
                        help="Continue training from --output if it already exists")
    parser.add_argument("--doors", type=str, default="1",
                        help="Comma-separated list of door indices (1..5) to "
                             "rotate through during training. Episodes cycle "
                             "through this list, so the resulting policy sees "
                             "every listed door pattern. Default '1' trains "
                             "against the leftmost-open layout only; pass "
                             "'1,2,3,4,5' to train on all five.")
    args = parser.parse_args()

    checkpoint_path = args.output.resolve()
    n_episodes = args.episodes

    try:
        door_list = [int(d.strip()) for d in args.doors.split(",") if d.strip()]
    except ValueError:
        sys.exit(f"[pretrain] --doors expects integers, got: {args.doors!r}")
    for d in door_list:
        if d not in (1, 2, 3, 4, 5):
            sys.exit(f"[pretrain] --doors entries must be in 1..5, got {d}")
        urdf = ROOT / "environment" / "linear_sunburst_map" / f"plane_doors_only_{d}.urdf"
        if not urdf.exists():
            sys.exit(f"[pretrain] missing URDF for door {d}: {urdf}\n"
                     f"           run: python environment/linear_sunburst_map/generate_door_variants.py")

    print("\n" + "=" * 70)
    print(f"  A2C PRE-TRAINING  —  {n_episodes} episodes")
    print(f"  output checkpoint: {checkpoint_path}")
    print("=" * 70)

    original_snapshot = ROOT / ".pretrain_a2c_original"
    snapshot_data(original_snapshot)
    print("[pretrain] Original data snapshot saved")

    init_start_stamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'=' * 60}")
    print(f"  INIT  [{init_start_stamp}]  exploration phase — from_data=false, plane_doors")
    print(f"  ({INIT_STEPS} scripted steps, typically 1–3 min wall-clock)")
    print(f"{'=' * 60}")

    import time as _time
    init_wall_start = _time.time()
    run_with_config(
        "INIT",
        cfg_override={
            "simulation": {
                "nr_steps": INIT_STEPS,
                "nr_steps_exploration": INIT_STEPS,
            },
            "grid_cell_network": {"from_data": False},
            "environment": {"doors_option": "plane_doors"},
        },
        quiet=False,  # Forward Progress lines so the user sees INIT is alive.
    )
    print(f"[pretrain] Init (exploration) phase complete in "
          f"{_time.time() - init_wall_start:.0f}s")

    init_snapshot = ROOT / ".pretrain_a2c_init"
    snapshot_data(init_snapshot)
    print("[pretrain] Init data snapshot saved")

    resume_from = checkpoint_path if (args.resume and checkpoint_path.exists()) else None
    if resume_from is not None:
        print(f"[pretrain] Resuming from existing checkpoint: {resume_from}")

    print(f"\n{'=' * 60}")
    print(f"  TRAINING  —  {n_episodes} episodes")
    print(f"{'=' * 60}")

    episode_results = []
    current_checkpoint = resume_from
    import time as _time

    for ep in range(1, n_episodes + 1):
        restore_data(init_snapshot)

        door_idx = door_list[(ep - 1) % len(door_list)]
        doors_option = f"plane_doors_only_{door_idx}"

        override = {
            "simulation": {
                "nr_steps": NAV_STEPS,
                "nr_steps_exploration": 0,
            },
            "grid_cell_network": {"from_data": True},
            "environment": {"doors_option": doors_option},
            "exploration": {"type": "a2c"},
            "thigmotaxis": {"enabled": True},
            "a2c_exploration": {
                "checkpoint_path": str(current_checkpoint) if current_checkpoint else None,
                "save_path": str(checkpoint_path),
                "frozen": False,
            },

            "plotting": {"live_plot": True},
        }

        ep_start_wall = _time.time()
        ep_start_stamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[pretrain {ep_start_stamp}] Episode {ep}/{n_episodes}  door={door_idx} "
              f"(loading from: {current_checkpoint or '<fresh>'})")
        print(f"[pretrain] -> running navigation (up to {NAV_STEPS} steps, max wall-clock varies)")

        metrics = run_with_config(f"Episode {ep}/{n_episodes}", override, quiet=False)
        episode_results.append(metrics)

        elapsed = _time.time() - ep_start_wall
        reached = metrics.get("goal_reached", False)
        nav = metrics.get("nav_steps")
        first_door = metrics.get("first_door_step", -1)
        unique_pcs = metrics.get("unique_pcs_visited", 0)
        final_err = metrics.get("final_error", float("nan"))
        outcome = (
            f"goal at nav step {nav}"
            if reached
            else f"TIMEOUT (final_error={final_err:.2f} m)"
        )
        print(f"[pretrain] Episode {ep}/{n_episodes} done in {elapsed:.0f}s  "
              f"door={door_idx}  outcome: {outcome}")
        print(f"[pretrain]   first_door={first_door}  unique_PCs={unique_pcs}  "
              f"nav_steps={nav if nav else 'n/a'}")
        if not checkpoint_path.exists():
            print(f"[pretrain] WARNING: episode {ep} did not save a checkpoint to "
                  f"{checkpoint_path}")
        else:
            cp_size_kb = checkpoint_path.stat().st_size / 1024
            print(f"[pretrain]   checkpoint updated: "
                  f"{checkpoint_path.name} ({cp_size_kb:.1f} KB)")
        current_checkpoint = checkpoint_path

    restore_data(original_snapshot)
    print("\n[pretrain] Original data restored")

    print(f"\n{'=' * 70}")
    print("  PRE-TRAINING SUMMARY")
    print(f"{'=' * 70}")

    success = [r for r in episode_results if r.get("goal_reached")]
    print(f"  Episodes:            {n_episodes}")
    print(f"  Successful episodes: {len(success)}/{n_episodes}")
    if success:
        nav = [r["nav_steps"] for r in success if r.get("nav_steps") is not None]
        if nav:
            arr = np.array(nav, dtype=float)
            print(f"  Nav steps (success): mean={np.mean(arr):.0f} std={np.std(arr):.0f}")
    print(f"  Final checkpoint:    {checkpoint_path}")
    print(f"{'=' * 70}\n")
    print("To evaluate this frozen policy, set in config.json:")
    print(f'  "a2c_exploration": {{')
    print(f'      "checkpoint_path": "{checkpoint_path.relative_to(ROOT)}",')
    print(f'      "frozen": true')
    print(f'  }}')
    print("Or pass --pretrained to benchmark_a2c.py once that flag is supported.\n")


if __name__ == "__main__":
    main()
