import os
import numpy as np
from plotting.plotResults import export_linear_lookahead_video
from plotting.plotThesis import plot_sub_goal_localization

# In-loop diagnostic plots (one PDF per lookahead call) are by far the
# heaviest non-physics cost in the simulation: each call constructs a fresh
# matplotlib figure, draws the cognitive map plus the agent's quiver, and
# writes a PDF to disk. Skip them entirely when BENCHMARK_NO_PLOTS=1 is set
# in the subprocess environment. The benchmark scripts opt into this; ad-hoc
# `python main.py` runs keep the plots for debugging.
_SKIP_LOOKAHEAD_PLOTS = os.environ.get("BENCHMARK_NO_PLOTS", "0") == "1"


def perform_lookahead_directed(gc_network, pc_network, cognitive_map, env):
    """Performs a linear lookahead in a preset direction"""
    gc_network.reset_s_virtual()  # Resets virtual gc spiking to actual spiking

    dt = gc_network.dt * 40  # checks spiking only every nth step
    speed = 0.5  # lookahead speed, becomes unstable for large speeds

    angles = np.linspace(0, 2 * np.pi, num=env.num_ray_dir, endpoint=False)  # lookahead directions

    goal_spiking = {}  # "angle": {"reward_value", "idx_place_cell", "distance", "step"}

    # Capped at 2.0 m so the directed lookahead only plans over the local
    # neighbourhood. Previously this was 0.5 * arena_size (~7.5 m), which let
    # the virtual rollout reach the goal PC from anywhere in the maze and
    # caused the lookahead to dominate route selection over physical
    # raycasting in tight quarters. The shorter horizon also reduces the
    # per-call cost of the lookahead (fewer virtual GC updates).
    max_distance = 3.0
    max_nr_steps = int(max_distance / (speed * dt))

    for idx, angle in enumerate(angles):

        # Check if lookahead direction is blocked
        if not env.directions[idx]:
            # If yes do not consider that direction
            goal_spiking[angle] = {"reward": -1, "idx_place_cell": -1,
                                   "distance": 0, "step": 0, "blocked": True}
            continue

        # Check if direction is one of the favored traveling directions
        if not idx % env.num_travel_dir == 0:
            # If no do not consider that direction
            goal_spiking[angle] = {"reward": -1, "idx_place_cell": -1,
                                   "distance": 0, "step": 0, "blocked": False}
            continue

        xy_speed = np.array([np.cos(angle), np.sin(angle)]) * speed  # lookahead velocity vector

        for i in range(max_nr_steps):
            firing_values = pc_network.compute_firing_values(gc_network.gc_modules, virtual=True)
            [reward, idx_place_cell] = cognitive_map.compute_reward_spiking(firing_values)  # highest reward spiking

            distance = np.linalg.norm(xy_speed * i * dt)  # lookahead distance traveled

            if angle not in goal_spiking or reward - goal_spiking[angle]["reward"] > 0:
                # First entrance or exceeds previous found value
                goal_spiking[angle] = {"reward": reward, "idx_place_cell": idx_place_cell,
                                       "distance": distance, "step": i, "blocked": False}

            # Abort conditions to end lookahead earlier
            if angle in goal_spiking and reward < 0.85 * goal_spiking[angle]["reward"] \
                    and goal_spiking[angle]["reward"] > 0.8 and i > 50:
                break

            gc_network.track_movement(xy_speed, virtual=True, dt_alternative=dt)  # track virtual movement

        gc_network.reset_s_virtual()  # reset after lookahead in a direction

    rewards = [a["reward"] for a in goal_spiking.values()]
    angle_keys = list(goal_spiking.keys())
    idx_angle = int(np.argmax(rewards))  # determine most promising direction

    # --- Hysteresis ---
    # Near the goal PC, several rollout directions all reach the goal PC at some
    # virtual step and return near-identical rewards. argmax then flips between
    # near-ties on every call, producing 180° goal-vector swings → the motor
    # controller spends most of its time rotating instead of translating.
    # If the previously chosen direction is still within HYSTERESIS_TOL of the
    # current best, keep it. Only switch when some other direction is clearly
    # better. Has no effect far from the goal where the reward gradient is sharp.
    HYSTERESIS_TOL = 0.05
    prev_idx = getattr(env, "_last_lookahead_idx_angle", None)
    if prev_idx is not None and 0 <= prev_idx < len(rewards) and prev_idx != idx_angle:
        best_reward = rewards[idx_angle]
        prev_reward = rewards[prev_idx]
        # Keep the previous direction if its reward is within tolerance of the
        # new best AND it's still positive (i.e. not blocked / -1).
        if prev_reward > 0 and best_reward > 0 and \
                (best_reward - prev_reward) <= HYSTERESIS_TOL * max(best_reward, 1e-3):
            idx_angle = prev_idx
    env._last_lookahead_idx_angle = idx_angle

    angle = angle_keys[idx_angle]
    reward = goal_spiking[angle]["reward"]

    # Cache best reward for navigationPhase to check
    env._last_lookahead_best_reward = max(reward, 0.0)

    # Stay in topology-based directed lookahead even when reward is high.
    # perform_look_ahead_2x can't see walls and produces bad vectors in mazes.
    if reward >= 0:
        distance = goal_spiking[angle]["distance"]
        distance = np.maximum(distance, 0.5) if reward < 0.8 else distance
        goal_vector = np.array([np.cos(angle), np.sin(angle)]) * distance  # goal vector to travel along
    else:
        goal_vector = np.random.rand(2) * 0.5

    if not _SKIP_LOOKAHEAD_PLOTS:
        filename = "_subgoal_" + str(len(env.xy_coordinates) - 1)
        plot_sub_goal_localization(env, cognitive_map, pc_network, env.goal_vector, filename, idx_angle, goal_spiking)

    return goal_vector
