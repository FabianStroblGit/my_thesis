# Bio-Inspired Navigation with Curiosity-Driven and Active-Inference Controllers

Code accompanying the Bachelor's thesis

> **Deep Reinforcement Learning for Curiosity-Based Navigation**
> Fabian Strobl — TUM, 2026

The project builds on the cognitive map of Engelmann (2021) — grid
cells, place cells, and prefrontal-cortex-like recency/topology/reward cells — and
adds three pluggable exploration controllers that take over when the stored
topology can no longer reach the goal (i.e. when a previously open door is closed
at evaluation time):

1. **A2C** — a pre-trained, frozen actor–critic policy with curiosity-style intrinsic rewards
2. **Active Inference (V2)** — a multi-step planner that minimises expected free energy over the place-cell graph
3. **Random** — a uniform-random baseline over unblocked directions

Each controller can additionally be paired with a **thigmotaxis** wall-following
module (an ethological prior). Evaluation happens in a `linear_sunburst`
PyBullet maze with five doors, only one of which is open at evaluation time.

For the full motivation, methods, and results, see the thesis PDF.

---

## Installation

The code is Python 3.13.1 and was developed on macOS / Linux.

```bash
pip install pybullet numpy scipy matplotlib torch
```

`ffmpeg` is bundled in `ffmpeg/` for optional video export.

---

## Quick start

```bash
# Run a single navigation episode using the controller selected in config.json
python main.py
```

The active controller is chosen by the `exploration.type` field in `config.json`:
- `"a2c"` — frozen A2C policy (requires a pre-trained checkpoint at `data/a2c_pretrained.pt`)
- `"active_inference"` (or `"active_inference_v2"`) — Active Inference planner
- `"random"` — uniform-random baseline

Thigmotaxis is toggled via `"thigmotaxis": {"enabled": true|false}`. The door
layout is set by `"environment": {"doors_option": "plane_doors_only_K"}` for
`K ∈ {1,2,3,4,5}` (the index of the single open door), or `"plane_doors"` for
the all-doors-open exploration phase.

---

```bash
python benchmark_a2c.py    --doors 1,2,3,4 --thigmo on --runs 1 --live
python benchmark_ai.py     --doors 1,2,3,4 --thigmo on --runs 1 --live
python benchmark_random.py --doors 1,2,3,4 --thigmo on --runs 1 --live
```

Each script:
1. Snapshots the current `data/` folder,
2. Runs the scripted exploration phase once (≈ 14 000 simulation steps, all doors open),
3. Snapshots the resulting cognitive map,
4. For every (door, thigmotaxis) cell, restores the post-exploration snapshot and runs N navigation episodes (each up to 150 000 steps, one door open),
5. Writes per-run trajectory plots to `benchmark_<x>_plots/<run_id>/` and an aggregate JSON to `benchmark_<x>_results_<run_id>.json`,
6. Restores the original `data/` folder.

Aggregation (mean ± std, success rate, etc.) is computed by the shared
`benchmark_summary.py` helper, imported by all three sweeps.

### A2C pre-training

The A2C policy used by `benchmark_a2c.py` must be pre-trained before
evaluation. It is **not** trained at evaluation time — the thesis design
freezes the policy after pre-training and measures generalisation across
door layouts.

```bash
python pretrain_a2c.py
```

This cycles through the four evaluation door layouts and writes
`data/a2c_pretrained.pt`.

### Figures

`generate_maze_figures.py` renders the maze layouts and the cognitive map after
the exploration phase (used for Figures 5.1 and 5.2 in the thesis). It requires
a `data/` folder populated by a prior `main.py` or benchmark run.

```bash
python generate_maze_figures.py
```

---

## Repository layout

```
.
├── main.py                          # Single-run entry point (see config.json)
├── pretrain_a2c.py                  # A2C pre-training orchestrator
├── benchmark_a2c.py                 # A2C sweep (used for thesis Tables 5.1–5.3)
├── benchmark_ai.py                  # Active Inference sweep
├── benchmark_random.py              # Random-walk baseline sweep
├── benchmark_summary.py             # Shared aggregation helper
├── generate_maze_figures.py         # Thesis-figure generator
├── config.json                      # Central configuration
│
├── system/
│   ├── bio_model/
│   │   ├── gridcellModel.py         # Grid-cell continuous-attractor modules
│   │   ├── placecellModel.py        # Place-cell network
│   │   ├── cognitivemapModel.py     # Topology + reward propagation
│   ├── controller/
│   │   ├── pybulletEnv.py           # PyBullet env + raycast sensing
│   │   ├── explorationPhase.py      # Scripted exploration trajectory
│   │   ├── navigationPhase.py       # Mode switching, thigmotaxis, lookahead
│   │   ├── a2cExplorer.py           # A2C actor–critic + intrinsic rewards
│   │   ├── activeInferenceV2.py     # Active Inference V2 planner
│   │   └── randomExplorer.py        # Uniform-random baseline
│   ├── decoder/
│   │   └── linearLookahead.py       # Directed grid-cell lookahead
│   └── helper.py                    # Math utilities
│
├── plotting/
│   ├── plotResults.py               # LiveCognitiveMapPlot + diagnostic plots
│   ├── plotThesis.py                # Static thesis-quality plot helpers
│   └── plotHelper.py                # Environment / robot drawing utilities
│
├── environment/                     # PyBullet URDF assets
│   └── linear_sunburst_map/         # Door-variant generator + URDFs
├── p3dx/                            # Pioneer 3DX robot URDF
├── data/                            # GC / PC / cognitive-map state (auto-populated)
├── logs/                            # Timestamped run logs
└── ffmpeg/                          # Bundled ffmpeg for optional video export
```

The folders `benchmark_a2c_plots/`, `benchmark_ai_plots/`,
`benchmark_random_plots/`, `experiments/`, `figures/`, and `plots/` are
auto-created when the respective scripts run.

---

## Configuration

`config.json` centralises everything that varies between experiments. The most
relevant fields:

| Field | Purpose |
|---|---|
| `simulation.dt` | Physics timestep (default `0.01 s`) |
| `simulation.nr_steps` | Step limit per navigation episode (default `150000`) |
| `grid_cell_network.modules` / `sheet_size` / `gmin` / `gmax` | Six 40×40 grid modules, scales 0.2 m–2.4 m |
| `grid_cell_network.from_data` | `true` = restore saved GC state; `false` = build from scratch |
| `environment.env_model` | `"linear_sunburst"` for the thesis maze |
| `environment.doors_option` | `"plane_doors"` (all open) or `"plane_doors_only_K"` (only door K open) |
| `exploration.type` | `"a2c"` / `"active_inference"` / `"random"` |
| `thigmotaxis.enabled` | Toggle the wall-following module |
| `a2c_exploration.*` | A2C learning rate, intrinsic-reward weights, checkpoint path |
| `active_inference_v2.*` | Anisotropy / momentum prior weights for the EFE planner |

---

## Acknowledgements

The base cognitive map (grid–place–PFC architecture, vector and
topology navigation) is adapted from Engelmann (2021). This thesis contributes
the three exploration controllers, the thigmotaxis module, and the
blocked-door evaluation protocol.
