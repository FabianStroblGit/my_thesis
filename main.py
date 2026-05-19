import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


import matplotlib
if os.environ.get("HEADLESS_MPL", "0") == "1":
    matplotlib.use("Agg")
    import warnings as _warnings
    _warnings.filterwarnings("ignore", message=".*non-interactive.*cannot be shown.*")

import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np
import pybullet as p

from plotting.plotResults import *
from plotting.plotResults import LiveCognitiveMapPlot
from system.bio_model.cognitivemapModel import CognitiveMapNetwork
from system.bio_model.gridcellModel import GridCellNetwork
from system.bio_model.placecellModel import PlaceCellNetwork
from system.controller.explorationPhase import compute_exploration_goal_vector
from system.controller.navigationPhase import compute_navigation_goal_vector
from system.controller.pybulletEnv import PybulletEnvironment
from system.controller.a2cExplorer import A2CExplorer, SpatialExplorationGrid, compute_intrinsic_reward
from system.controller.activeInferenceV2 import ActiveInferenceV2Explorer
from system.controller.randomExplorer import RandomExplorer



# --- Logging Setup ---
class TeeStream:
    """A stream that writes to both a file and original stream."""
    def __init__(self, file_stream, original_stream):
        self.file_stream = file_stream
        self.original_stream = original_stream
    
    def write(self, message):
        self.original_stream.write(message)
        self.file_stream.write(message)
        self.file_stream.flush()  # Ensure immediate write
    
    def flush(self):
        self.original_stream.flush()
        self.file_stream.flush()


def setup_logging():
    """Set up logging to capture all output to a timestamped log file."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"run_{timestamp}.log"
    
    # Open log file
    log_file_handle = open(log_file, 'w', encoding='utf-8')
    
    # Redirect stdout and stderr to both console and file
    sys.stdout = TeeStream(log_file_handle, sys.__stdout__)
    sys.stderr = TeeStream(log_file_handle, sys.__stderr__)
    
    print(f"=== Simulation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"=== Log file: {log_file.absolute()} ===\n")
    
    return log_file_handle, log_file


# Initialize logging
log_handle, log_path = setup_logging()


CONFIG_PATH = Path(__file__).with_name("config.json")
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    config = json.load(config_file)

paths_config = config["paths"]
simulation_config = config["simulation"]
grid_cell_config = config["grid_cell_network"]

_seed = simulation_config.get("seed")
if _seed is not None:
    import random as _random
    np.random.seed(int(_seed))
    _random.seed(int(_seed))
    print(f"[seed] numpy + random seeded with {int(_seed)}")
environment_config = config["environment"]
phase_offset_config = config["phase_offset_detector"]
video_config = config["video"]
plotting_config = config["plotting"]
camera_config = config.get("camera", {})


mpl.rcParams['animation.ffmpeg_path'] = paths_config["ffmpeg_path"]

# capture frequently tweaked parameters from configuration
dt = simulation_config["dt"]

M = grid_cell_config["modules"]
n = grid_cell_config["sheet_size"]
gmin = grid_cell_config["gmin"]
gmax = grid_cell_config["gmax"]
from_data = grid_cell_config["from_data"]

visualize = environment_config["visualize"]
env_model = environment_config["env_model"]
vector_model = environment_config["vector_model"]
doors_option = environment_config.get("doors_option", "plane_doors")

phase_bins = phase_offset_config["phase_bins"]
speed_bins = phase_offset_config["speed_bins"]

video = video_config["enabled"]
fps = video_config["fps"]
step = max(1, int((1 / max(fps, 1)) / dt))

plot_matching_vectors = plotting_config["plot_matching_vectors"]

video_output_dir = Path(paths_config["video_output_dir"])
experiment_output_dir = Path(paths_config["experiment_output_dir"])

gc_network = GridCellNetwork(n, M, dt, gmin, gmax=gmax, from_data=from_data)

pod_network = None
spike_detector = None

env = PybulletEnvironment(visualize, env_model, dt, pod=pod_network, doors_option=doors_option,
                          camera_config=camera_config)

# Wire thigmotaxis config flag to env
env.thigmotaxis_enabled = config.get("thigmotaxis", {}).get("enabled", True)

env.skip_left_lower_init = bool(config.get("exploration", {}).get("skip_left_lower_init", False))


# initialize Place Cell Model
pc_network = PlaceCellNetwork(from_data=from_data)
# initialize Cognitive Map Model
cognitive_map = CognitiveMapNetwork(dt, from_data=from_data)

if from_data:
    # Find the PC nearest to the actual goal location and make it the reward source.
    # This fixes reward propagation assigning the highest reward to the wrong PC.
    goal_pc_idx = cognitive_map.set_goal_pc_by_location(env.goal_location, pc_network)
    if goal_pc_idx is not None:
        goal_pc_pos = pc_network.place_cells[goal_pc_idx].env_coordinates
        print(f"Set goal PC{goal_pc_idx} at ({goal_pc_pos[0]:.1f},{goal_pc_pos[1]:.1f}), "
              f"goal_location=({env.goal_location[0]:.1f},{env.goal_location[1]:.1f})")
        gc_network.set_as_target_state(pc_network.place_cells[goal_pc_idx].gc_connections)
    else:
        idx = np.argmax(cognitive_map.reward_cells)
        gc_network.set_as_target_state(pc_network.place_cells[idx].gc_connections)

# Initialize exploration strategy (Active Inference or A2C)
a2c_config = config.get("a2c_exploration", {})
explorer_type = config.get("exploration", {}).get("type", "active_inference")

spatial_grid = SpatialExplorationGrid(
    arena_size=environment_config.get("arena_size", 15),
    cell_size=a2c_config.get("cell_size", 0.5)
)

# Both "active_inference" and "active_inference_v2" select the V2 planner.
# The legacy v1 greedy planner was removed; the alias is preserved so
# existing configs and benchmark scripts that pass "active_inference"
# continue to work.
if explorer_type in ("active_inference", "active_inference_v2"):
    # Multi-step Active Inference planner over the cognitive-map topology.
    # See system/controller/activeInferenceV2.py for the formulation.
    ai_v2_config = config.get("active_inference_v2", {})
    ai_explorer = ActiveInferenceV2Explorer(
        cognitive_map=cognitive_map,
        pc_network=pc_network,
        gc_network=gc_network,
        horizon=ai_v2_config.get("horizon", 3),
        branching=ai_v2_config.get("branching", 4),
        epistemic_weight=ai_v2_config.get("epistemic_weight", 1.0),
        extrinsic_weight=ai_v2_config.get("extrinsic_weight", 2.0),
        inverse_temperature=ai_v2_config.get("inverse_temperature", 2.0),
        temporal_discount=ai_v2_config.get("temporal_discount", 0.9),
        action_repeat=ai_v2_config.get("action_repeat", 200),
        momentum_weight=ai_v2_config.get("momentum_weight", 0.2),
        wall_margin=ai_v2_config.get("wall_margin", 0.5),
        anisotropy_weight=ai_v2_config.get("anisotropy_weight", 1.5),
        virtual_min_real_dist=ai_v2_config.get("virtual_min_real_dist", 0.8),
    )
    a2c_explorer = None
    a2c_transitions = []
    print(f"[AI2] Initialized Active Inference v2 explorer "
          f"(horizon={ai_v2_config.get('horizon', 3)}, "
          f"branching={ai_v2_config.get('branching', 4)}, "
          f"epistemic={ai_v2_config.get('epistemic_weight', 1.0)}, "
          f"extrinsic={ai_v2_config.get('extrinsic_weight', 2.0)})")
elif explorer_type == "random":
    # Random-walk baseline: same select_action API as the AIF explorer so it
    # plugs into the same goal-vector code path in navigationPhase.
    rnd_config = config.get("random_exploration", {})
    ai_explorer = RandomExplorer(
        num_directions=16,
        action_repeat=rnd_config.get("action_repeat", 200),
        wall_margin=rnd_config.get("wall_margin", 0.3),
        rng_seed=rnd_config.get("rng_seed", None),
    )
    a2c_explorer = None
    a2c_transitions = []
    print(f"[RND] Initialized Random-walk explorer "
          f"(action_repeat={rnd_config.get('action_repeat', 200)}, "
          f"seed={rnd_config.get('rng_seed', None)})")
else:
    ai_explorer = None
    a2c_explorer = A2CExplorer(
        state_dim=13, action_dim=5,
        lr=a2c_config.get("learning_rate", 0.0003),
        gamma=a2c_config.get("gamma", 0.99),
        entropy_coef=a2c_config.get("entropy_coef", 0.01)
    )
    a2c_transitions = []
    print(f"[A2C] Initialized A2C explorer (lr={a2c_config.get('learning_rate', 0.0003)}, action_repeat=300)")

# A2C policy is frozen during the navigation phase when this flag is set
# (no `update()` calls). Used for evaluation of a pre-trained checkpoint.
a2c_frozen = bool(a2c_config.get("frozen", False))

# Load a pre-trained A2C checkpoint if requested.
a2c_checkpoint = a2c_config.get("checkpoint_path") or None
if a2c_explorer is not None and a2c_checkpoint:
    ckpt_path = Path(a2c_checkpoint)
    if ckpt_path.exists():
        try:
            a2c_explorer.load(str(ckpt_path))
            print(f"[A2C] Loaded pre-trained checkpoint from {ckpt_path} "
                  f"(training_steps={a2c_explorer.training_steps}, frozen={a2c_frozen})")
        except Exception as exc:
            print(f"[A2C] WARNING: failed to load checkpoint {ckpt_path}: {exc}")
    else:
        print(f"[A2C] WARNING: checkpoint_path {ckpt_path} not found — starting fresh")


# run simulation
nr_steps = simulation_config["nr_steps"]
nr_steps_exploration = simulation_config["nr_steps_exploration"]
nr_plots = simulation_config["nr_plots"]
nr_trials = simulation_config["nr_trials"]

if video:
    [fig, f_gc, f_t, f_mon] = layout_video()
    live_plot = None
else:
    fig = None
    # Initialize live cognitive map visualization if enabled
    live_plot_enabled = plotting_config.get("live_plot", True)
    live_plot_interval = plotting_config.get("live_plot_interval", 50)
    if live_plot_enabled:
        live_plot = LiveCognitiveMapPlot(
            environment=env_model,
            door_positions=getattr(env, "door_positions", None),
            update_interval=live_plot_interval
        )
        print(f"[LIVE PLOT] Initialized live cognitive map visualization (interval={live_plot_interval})")
    else:
        live_plot = None

# Save across frames
goal_vector_array = [np.array([0, 0])]  # array to save the calculated goal vector

# --- Metric tracking for benchmarks ---
_metrics = {
    "wall_hug_steps": 0,       # steps with nearest wall <= 0.4m
    "nav_steps_total": 0,      # total navigation steps executed
    "unique_pcs": set(),        # distinct PC indices visited during navigation
    "door_traversals": [],      # list of (step, door_x) when agent crosses dividing wall
    "prev_y": None,             # previous y-coordinate for door crossing detection
    "goal_step": None,          # step at which goal was reached
}
WALL_Y = 5.4  # approximate y-coordinate of the dividing wall


def _handle_a2c_transition(env_ref, cognitive_map_ref, spatial_grid_ref, current_step):
    """Handle A2C transition: compute intrinsic reward, store, and train periodically."""
    if env_ref.pending_transition is not None and env_ref.exploration_mode:
        pos = env_ref.xy_coordinates[-1]
        spatial_grid_ref.visit(pos[0], pos[1], current_step)
        
        # Compute intrinsic reward from PC-based novelty
        current_pc = getattr(env_ref, 'current_pc_idx', None)
        if current_pc is not None and current_pc < len(cognitive_map_ref.visit_counts):
            vc = cognitive_map_ref.visit_counts[current_pc]
            lvs = cognitive_map_ref.last_visit_step[current_pc]
        else:
            vc = 0
            lvs = -1
        reward = compute_intrinsic_reward(vc, lvs, current_step)
        env_ref.pending_transition['reward'] = reward
        a2c_transitions.append(env_ref.pending_transition)
        env_ref.pending_transition = None
        
        # Train A2C every 10 transitions (skipped when policy is frozen,
        # e.g. when evaluating a pre-trained checkpoint).
        if len(a2c_transitions) >= 10:
            if not a2c_frozen:
                stats = a2c_explorer.update(a2c_transitions)
            a2c_transitions.clear()



# this function performs the simulation steps and is called by video creator or manually
def animation_frame(frame):
    if video:
        # calculate how many simulations steps to do for current frame
        start = frame - step
        end = frame
        if start < 0:
            start = 0
    else:
        # run trough all simulation steps as no frames have to be exported
        start = 0
        end = frame

    for i in range(start, end):
        # perform one simulation step
        exploration_phase = True if i < nr_steps_exploration else False

        # Per-block timing instrumentation. Enabled when PROFILE_STEPS is
        # set (env var) — prints accumulated time per block every N steps.
        _prof_enabled = os.environ.get("PROFILE_STEPS", "1") == "1"
        if _prof_enabled:
            import time as _t
            if "_prof" not in dir():
                _prof = {"goal_vec": 0.0, "move": 0.0, "gc": 0.0, "pc": 0.0,
                         "cog_map": 0.0, "prune": 0.0, "a2c_tx": 0.0, "n": 0}
            _t0 = _t.perf_counter()

        # compute goal vector
        if exploration_phase:
            compute_exploration_goal_vector(env, i)
        else:
            compute_navigation_goal_vector(gc_network, pc_network, cognitive_map, i - nr_steps_exploration, env,
                                           pod=pod_network, spike_detector=spike_detector, model=vector_model,
                                           a2c_explorer=a2c_explorer, spatial_grid=spatial_grid,
                                           ai_explorer=ai_explorer)
        goal_vector_array.append(env.goal_vector)
        if _prof_enabled:
            _t1 = _t.perf_counter(); _prof["goal_vec"] += _t1 - _t0; _t0 = _t1

        # compute velocity vector
        env.compute_movement(gc_network, pc_network, cognitive_map, exploration_phase=exploration_phase)
        xy_speed = env.xy_speeds[-1]
        if _prof_enabled:
            _t1 = _t.perf_counter(); _prof["move"] += _t1 - _t0; _t0 = _t1

        # grid cell network track movement
        gc_network.track_movement(xy_speed)
        if _prof_enabled:
            _t1 = _t.perf_counter(); _prof["gc"] += _t1 - _t0; _t0 = _t1

        # place cell network track gc firing
        goal_distance = np.linalg.norm(env.xy_coordinates[-1] - env.goal_location)
        reward = 1 if goal_distance < 0.1 else 0
        reward_first_found = False
        if reward == 1 and (len(cognitive_map.reward_cells) == 0 or np.max(cognitive_map.reward_cells) != 1):
            reward_first_found = True
            gc_network.set_current_as_target_state()

        [firing_values, created_new_pc] = pc_network.track_movement(
            gc_network.gc_modules, reward_first_found)

        if created_new_pc:
            pc_network.place_cells[-1].env_coordinates = np.array(env.xy_coordinates[-1])
        if _prof_enabled:
            _t1 = _t.perf_counter(); _prof["pc"] += _t1 - _t0; _t0 = _t1

        # cognitive map track pc firing
        # Cap edge length at 2.0 m to prevent ghost shortcuts through
        # walls under non-smooth motion (wall-follow, retreat, door
        # traversal); temporal recency window still applies.
        cognitive_map.track_movement(firing_values, created_new_pc, reward, env=env, current_step=i,
                                    pc_network=pc_network, max_connection_dist=2.0)
        if _prof_enabled:
            _t1 = _t.perf_counter(); _prof["cog_map"] += _t1 - _t0; _t0 = _t1

        # Prune topology connections blocked by walls (only during navigation)
        # Wait 100 steps for directions to stabilize, then prune every 50 steps
        nav_step = i - nr_steps_exploration
        if (not exploration_phase and nav_step >= 100 and nav_step % 50 == 0
                and getattr(env, 'current_pc_idx', None) is not None):
            cognitive_map.prune_blocked_connections(
                env.current_pc_idx, np.array(env.xy_coordinates[-1]),
                env.directions, pc_network)
        if _prof_enabled:
            _t1 = _t.perf_counter(); _prof["prune"] += _t1 - _t0; _t0 = _t1

        # Handle A2C transition (intrinsic reward + training)
        _handle_a2c_transition(env, cognitive_map, spatial_grid, i)
        if _prof_enabled:
            _t1 = _t.perf_counter(); _prof["a2c_tx"] += _t1 - _t0
            _prof["n"] += 1
            if _prof["n"] % 5000 == 0:
                total = sum(_prof[k] for k in _prof if k != "n")
                print(f"[PROF] step={i} n={_prof['n']} total={total:.2f}s  "
                      f"goal_vec={_prof['goal_vec']:.2f} ({100*_prof['goal_vec']/total:.0f}%)  "
                      f"move={_prof['move']:.2f} ({100*_prof['move']/total:.0f}%)  "
                      f"gc={_prof['gc']:.2f} ({100*_prof['gc']/total:.0f}%)  "
                      f"pc={_prof['pc']:.2f} ({100*_prof['pc']/total:.0f}%)  "
                      f"cog_map={_prof['cog_map']:.2f} ({100*_prof['cog_map']/total:.0f}%)  "
                      f"prune={_prof['prune']:.2f} ({100*_prof['prune']/total:.0f}%)  "
                      f"a2c_tx={_prof['a2c_tx']:.2f} ({100*_prof['a2c_tx']/total:.0f}%)  "
                      f"PCs={cognitive_map.nr_place_cells}")

        # --- Collect metrics during navigation phase ---
        if not exploration_phase:
            _metrics["nav_steps_total"] += 1
            pos = env.xy_coordinates[-1]

            # Wall-hugging: use last_min_ray_dist cached by avoid_obstacles (no extra raycast)
            min_ray = getattr(env, '_last_min_ray_dist', 2.0)
            if min_ray <= 0.4:
                _metrics["wall_hug_steps"] += 1

            # Unique PCs visited
            pc_idx = getattr(env, 'current_pc_idx', None)
            if pc_idx is not None:
                _metrics["unique_pcs"].add(pc_idx)

            # Door traversal detection (crossing y ≈ WALL_Y)
            cur_y = float(pos[1])
            prev_y = _metrics["prev_y"]
            if prev_y is not None:
                if (prev_y < WALL_Y and cur_y >= WALL_Y) or (prev_y >= WALL_Y and cur_y < WALL_Y):
                    door_x = float(pos[0])
                    _metrics["door_traversals"].append((i, round(door_x, 1)))
                    if len(_metrics["door_traversals"]) == 1:
                        print(f"[METRIC] First door traversal at step {i}, x={door_x:.1f}")
            _metrics["prev_y"] = cur_y

        # Check if goal was reached during navigation phase — end simulation early.
        # Radius bumped from 0.5 m to 1.0 m so that the agent isn't trapped in
        # the goal-alley oscillation pattern (lookahead direction flips between
        # east/west at the top wall, ping-pong positions of ~0.6-0.9 m from
        # goal). 1.0 m captures the "the agent has effectively arrived" zone.
        GOAL_RADIUS = 1.0
        if not exploration_phase:
            any_reached = np.linalg.norm(env.xy_coordinates[-1] - env.goal_location) < GOAL_RADIUS
            if any_reached:
                _metrics["goal_step"] = i
                print(f"[GOAL] Goal reached at step {i}! Ending simulation.")
                break

        # plot or print intermediate update in console
        if not video and i % int(nr_steps / nr_plots) == 0:
            progress_str = "Progress: " + str(int(i * 100 / nr_steps)) + "%"
            print(progress_str)
            # plotCurrentAndTarget(gc_network.gc_modules)
        
        # Live cognitive map visualization (every update_interval steps)
        if live_plot is not None:
            live_plot.update(pc_network, cognitive_map,
                            xy_coordinates=env.xy_coordinates,
                            step=i)

    # simulated steps until next frame
    if video:
        # export current state as frame
        exploration_phase = True if frame < nr_steps_exploration else False
        plot_current_state(env, gc_network.gc_modules, f_gc, f_t, f_mon,
                           pc_network=pc_network, cognitive_map=cognitive_map,
                           exploration_phase=exploration_phase, goal_vector=goal_vector_array[-1])
        progress_str = "Progress: " + str(int((frame * 100) / nr_steps)) + "% | Current video is: " + str(
            frame * dt) + "s long"
        print(progress_str)


if video:
    # initialize video and call simulation function within
    frames = np.arange(0, nr_steps, step)
    anim = animation.FuncAnimation(fig, func=animation_frame, frames=frames, interval=1 / fps, blit=False)

    # Finished simulation

    # Export video
    video_output_dir.mkdir(parents=True, exist_ok=True)

    f = video_output_dir / "animation.mp4"
    video_writer = animation.FFMpegWriter(fps=fps)
    anim.save(str(f), writer=video_writer)
    env.end_simulation()
else:
    # manually call simulation function
    animation_frame(nr_steps)

    # Finished simulation

    # Plot last state (save to file if BENCHMARK_PLOT_PATH is set, otherwise show)
    _plot_save_path = os.environ.get("BENCHMARK_PLOT_PATH")
    cognitive_map_plot(pc_network, cognitive_map, xy_coordinates=env.xy_coordinates,
                       environment=env_model,
                       door_positions=getattr(env, "door_positions", None),
                       save_path=_plot_save_path)

    # Save place network and cognitive map to reload it later
    pc_network.save_pc_network()  # provide filename="_navigation" to avoid overwriting the exploration phase
    cognitive_map.save_cognitive_map()  # provide filename="_navigation" to avoid overwriting the exploration phase

    # Persist A2C model after the run if requested (used by pretrain_a2c.py
    # to accumulate weights across episodes).
    a2c_save_target = a2c_config.get("save_path") or None
    if a2c_explorer is not None and a2c_save_target:
        save_path = Path(a2c_save_target)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            a2c_explorer.save(str(save_path))
            print(f"[A2C] Saved checkpoint to {save_path} "
                  f"(training_steps={a2c_explorer.training_steps})")
        except Exception as exc:
            print(f"[A2C] WARNING: failed to save checkpoint to {save_path}: {exc}")

    # Calculate the distance between goal and actual end position (only relevant for navigation phase)
    error = np.linalg.norm((env.xy_coordinates[-1] + env.goal_vector) - env.goal_location)
    env.end_simulation()  # disconnect pybullet

    # --- Emit structured metrics for benchmark parsing ---
    nav_total = _metrics["nav_steps_total"]
    wall_hug_frac = _metrics["wall_hug_steps"] / max(nav_total, 1)
    unique_pcs = len(_metrics["unique_pcs"])
    door_trav = _metrics["door_traversals"]
    first_door_step = door_trav[0][0] if door_trav else -1
    num_door_crossings = len(door_trav)
    goal_step = _metrics["goal_step"]
    metrics_dict = {
        "goal_reached": goal_step is not None,
        "goal_step": goal_step,
        "nav_steps": (goal_step - nr_steps_exploration) if goal_step is not None else None,
        "nav_steps_total": nav_total,
        "wall_hug_fraction": round(wall_hug_frac, 4),
        "unique_pcs_visited": unique_pcs,
        "first_door_step": first_door_step,
        "num_door_crossings": num_door_crossings,
        "final_error": round(float(error), 4),
    }
    print(f"[METRICS] {json.dumps(metrics_dict)}")

    # Data to save to perform analysis later on
    error_array = [error]
    gc_array = [gc_network.consolidate_gc_spiking()]
    position_array = [env.xy_coordinates]
    vector_array = [goal_vector_array]

    progress_str = "Progress: " + str(int(1 * 100 / nr_trials)) + "% | Latest error: " + str(error)
    print(progress_str)

    # for the decoder test several trials are performed one after each other
    for i in range(1, nr_trials):
        gc_network.load_initialized_network("s_vectors_initialized.npy")
        pc_network = PlaceCellNetwork()
        cognitive_map = CognitiveMapNetwork(dt)
        env = PybulletEnvironment(visualize, env_model, dt, pod=pod_network, doors_option=doors_option,
                      camera_config=camera_config)
        env.thigmotaxis_enabled = config.get("thigmotaxis", {}).get("enabled", True)

        goal_vector_array = [np.array([0, 0])]

        animation_frame(nr_steps)
        error = np.linalg.norm((env.xy_coordinates[-1] + env.goal_vector) - env.goal_location)

        error_array.append(error)
        gc_array.append(gc_network.consolidate_gc_spiking())
        position_array.append(env.xy_coordinates)
        vector_array.append(goal_vector_array)

        env.end_simulation()

        progress_str = "Progress: " + str(int((i + 1) * 100 / nr_trials)) + "% | Latest error: " + str(error)
        print(progress_str)

    # Directly plot and print the errors (distance between goal and actual end position)
    # Filter out NaN values before plotting
    error_array_clean = [e for e in error_array if not np.isnan(e)]
    if len(error_array_clean) > 0:
        error_plot(error_array_clean)
    else:
        print("[WARNING] All error values are NaN, skipping error plot")
    # Save the data of all trials in a dedicated folder
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    np.save(experiment_output_dir / "error_array", error_array)
    np.save(experiment_output_dir / "gc_array", gc_array)
    np.save(experiment_output_dir / "position_array", position_array)
    np.save(experiment_output_dir / "vectors_array", vector_array)

    # --- Finalize logging ---
    print(f"\n=== Simulation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"=== Log saved to: {log_path.absolute()} ===")
    
    # Restore original streams and close log file
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_handle.close()

