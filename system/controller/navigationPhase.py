import numpy as np
import pybullet as p
from collections import deque
from system.decoder.linearLookahead import *
from system.controller.a2cExplorer import build_exploration_state, action_to_goal_vector


_NEAR_GOAL_HOPS = 2


def _get_ray_distances_toward(env, angles, ray_len=2.0):
    """Cast rays at the given world-frame angles and return distances.

    Wrapper around PyBullet raycasting. ``ray_len`` defaults to 2 m for
    the wall-proximity / wall-follow paths; the door scanner passes a
    longer length to distinguish ray-hits-wall from ray-passes-through-door.
    """
    ray_from_point = np.array(p.getLinkState(env.carID, 0)[0])
    ray_from_point[2] += 0.02

    ray_from = []
    ray_to = []
    for angle in angles:
        ray_from.append(ray_from_point)
        ray_to.append(np.array([
            np.cos(angle) * ray_len + ray_from_point[0],
            np.sin(angle) * ray_len + ray_from_point[1],
            ray_from_point[2]
        ]))

    ray_dist = np.full(len(angles), ray_len)
    results = p.rayTestBatch(ray_from, ray_to, numThreads=0)
    for idx, result in enumerate(results):
        if result[0] != -1:
            hit_position = result[3]
            ray_dist[idx] = np.linalg.norm(np.array(hit_position) - ray_from_point)

    return ray_dist


def _goal_angle(env):
    """Compute the world-frame angle from the agent to the goal."""
    pos = env.xy_coordinates[-1] if env.xy_coordinates else None
    goal_loc = env.goal_location if hasattr(env, 'goal_location') else None
    if pos is None or goal_loc is None:
        return 0.0
    diff = np.array(goal_loc) - np.array(pos)
    return np.arctan2(diff[1], diff[0])


def _walls_nearby(env, threshold=1.5):
    """True if at least one of 16 probe rays hits a surface within threshold."""
    probe_angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    probe_dist = _get_ray_distances_toward(env, probe_angles)
    return any(d < threshold for d in probe_dist)


def _is_frustrated(env, progress_window=300, min_progress=0.3):
    """Check if the agent is failing to make progress toward the goal.

    Returns: (frustrated: bool, goal_angle: float)
    """
    pos = env.xy_coordinates[-1] if env.xy_coordinates else None
    goal_loc = env.goal_location if hasattr(env, 'goal_location') else None
    if pos is None or goal_loc is None:
        return False, 0.0

    diff = np.array(goal_loc) - np.array(pos)
    goal_angle = np.arctan2(diff[1], diff[0])
    current_dist = np.linalg.norm(diff)

    if len(env.xy_coordinates) < progress_window:
        return False, goal_angle

    old_pos = np.array(env.xy_coordinates[-progress_window])
    old_dist = np.linalg.norm(np.array(goal_loc) - old_pos)
    progress = old_dist - current_dist  # positive = getting closer

    cooldown_remaining = getattr(env, '_door_cooldown', 0)
    if cooldown_remaining > 0:
        env._door_cooldown = cooldown_remaining - 1
        return False, goal_angle

    last_door_pos = getattr(env, '_last_door_pos', None)
    if last_door_pos is not None:
        dist_from_door = np.linalg.norm(np.array(pos) - np.array(last_door_pos))
        if dist_from_door < 1.0:
            # Still near the same door — suppress frustration so the agent moves away first.
            env._last_door_pos = None
            env._door_cooldown = progress_window
            return False, goal_angle

    frustrated = progress < min_progress

    if getattr(env, '_has_path', False):
        hist = getattr(env, '_graph_dist_history', None)
        cur_dist = getattr(env, '_graph_dist_to_goal', None)
        if hist and cur_dist is not None and len(hist) >= 2:
            old_dist = hist[0]
            if cur_dist < old_dist:
                # Hop count to goal decreased; not frustrated regardless of Euclidean stall.
                return False, goal_angle
    return frustrated, goal_angle


def _detect_door_opening(env, goal_angle):
    """Detect a real door in a north-anchored fan.

    Specialised for the linear_sunburst maze: doors sit in a single
    east-west partition wall, and the agent searches from south toward
    the goal in the upper region. Sweeps a 17-ray fan spanning NORTH +- 60
    degrees (120 total) and looks for the SHORT (pillar) -> LONG (doorway)
    -> SHORT (pillar) pattern.

    Close-range mode: when min(fan_dist) <= 1.2 m the agent is hugging a
    wall, so any LONG run is a doorway and the bounding requirement is dropped.

    Returns: np.array(2,) unit direction through the gap, or None.
    """
    NORTH = np.pi / 2
    n_fan = 17
    fan_spread = np.pi / 3  # half-angle; total fan = 120 degrees
    fan_angles = [NORTH + fan_spread * (i - n_fan // 2) / (n_fan // 2)
                  for i in range(n_fan)]
    DOOR_RAY_LEN = 5.0
    fan_dist = _get_ray_distances_toward(env, fan_angles, ray_len=DOOR_RAY_LEN)

    door_clear_threshold = 3.5
    pillar_max_clear = 1.5
    max_door_rays = 8   # > this = arena edge or whole-fan-open, not a door
    min_door_rays = 1

    wall_contact_dist = 1.2
    close_range = float(np.min(fan_dist)) <= wall_contact_dist

    is_long = [d >= door_clear_threshold for d in fan_dist]
    is_short = [d <= pillar_max_clear for d in fan_dist]

    pos = np.array(env.xy_coordinates[-1]) if env.xy_coordinates else None
    wf_steps = getattr(env, 'wall_follow_steps', 0)

    # Find maximal contiguous LONG runs in the fan.
    runs = []
    i = 0
    while i < n_fan:
        if is_long[i]:
            j = i
            while j < n_fan and is_long[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1

    # Filter runs to door-shaped ones.
    width_cap = n_fan if close_range else max_door_rays
    door_runs = []
    for left, right in runs:
        width = right - left + 1
        if width < min_door_rays or width > width_cap:
            continue
        left_bound = (left > 0 and is_short[left - 1])
        right_bound = (right < n_fan - 1 and is_short[right + 1])
        if close_range:
            pass
        elif left_bound and right_bound:
            pass
        else:
            continue
        mean_clear = float(np.mean([fan_dist[k] for k in range(left, right + 1)]))
        door_runs.append((left, right, width, mean_clear))

    if wf_steps % 50 == 0 and pos is not None:
        n_long = sum(is_long)
        n_short = sum(is_short)
        max_clear = float(max(fan_dist)) if len(fan_dist) > 0 else 0.0
        min_clear = float(min(fan_dist)) if len(fan_dist) > 0 else 0.0
        print(f"[THIGMO] door scan: n_long={n_long}/{n_fan}, "
              f"n_short={n_short}/{n_fan}, runs={len(runs)}, "
              f"door_runs={len(door_runs)}, "
              f"max_clear={max_clear:.2f}m, min_clear={min_clear:.2f}m, "
              f"close={close_range}, "
              f"pos=({pos[0]:.1f},{pos[1]:.1f}), wf_step={wf_steps}")

    if not door_runs:
        return None

    # Pick the run with highest mean clearance (deepest doorway).
    door_runs.sort(key=lambda r: -r[3])
    left, right, width, mean_clear = door_runs[0]

    # Reject if too close to a previously-blocked door location.
    BLOCKED_DOOR_RADIUS = 0.5
    blocked_doors = getattr(env, '_blocked_door_positions', [])
    if pos is not None:
        for blocked_pos in blocked_doors:
            if np.linalg.norm(pos - blocked_pos) < BLOCKED_DOOR_RADIUS:
                if wf_steps % 50 == 0:
                    print(f"[THIGMO] door scan: gap_found but REJECTED "
                          f"(within {BLOCKED_DOOR_RADIUS}m of previously-blocked "
                          f"door at ({blocked_pos[0]:.1f},{blocked_pos[1]:.1f}))")
                return None

    # Clearance-weighted centroid of the door run.
    gap_vec = np.array([0.0, 0.0])
    for k in range(left, right + 1):
        a = fan_angles[k]
        d = fan_dist[k]
        gap_vec += np.array([np.cos(a), np.sin(a)]) * d
    norm = float(np.linalg.norm(gap_vec))
    if norm > 0.01:
        gap_vec = gap_vec / norm
    else:
        center = fan_angles[(left + right) // 2]
        gap_vec = np.array([np.cos(center), np.sin(center)])

    print(f"[THIGMO] DOOR FOUND at pos=({pos[0]:.1f},{pos[1]:.1f}), "
          f"run=rays[{left}..{right}] width={width}, "
          f"mean_clear={mean_clear:.2f}m, "
          f"gap_dir=({gap_vec[0]:+.2f},{gap_vec[1]:+.2f}), "
          f"wf_step={wf_steps}")

    return gap_vec


def _compute_wall_follow_vector(env, goal_angle, nr_steps):
    """Compute goal vector for wall-following (thigmotaxis).

    Of all directions that are free (no whisker contact), pick the one
    most aligned with the goal.
    """
    probe_angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    probe_dist = _get_ray_distances_toward(env, probe_angles)

    pos = np.array(env.xy_coordinates[-1])
    goal_loc = np.array(env.goal_location)
    goal_diff = goal_loc - pos
    goal_norm = np.linalg.norm(goal_diff)
    goal_dir = goal_diff / goal_norm if goal_norm > 0.01 else np.array([0.0, 0.0])

    free_threshold = 1.0  # whisker length — blocked if wall closer than this
    best_score = -999
    best_vec = None

    for i, angle in enumerate(probe_angles):
        direction = np.array([np.cos(angle), np.sin(angle)])

        if probe_dist[i] < free_threshold:
            continue

        # Score by goal alignment (modulated by wall_follow_direction for reversals)
        score = env.wall_follow_direction * np.dot(direction, goal_dir)

        if score > best_score:
            best_score = score
            best_vec = direction

    if best_vec is None:
        # All directions blocked — push away from all walls.
        repulsion = np.array([0.0, 0.0])
        for i, angle in enumerate(probe_angles):
            wall_dir = np.array([np.cos(angle), np.sin(angle)])
            repulsion -= wall_dir / max(probe_dist[i], 0.1)
        norm = np.linalg.norm(repulsion)
        if norm > 0.01:
            best_vec = repulsion / norm
        else:
            best_vec = goal_dir

    return best_vec


def _compute_thigmotaxis_vector(env, nr_steps):
    """Wall-following with door detection.

    Returns: (goal_vector unit vector, is_active bool).
    """
    reverse_steps = getattr(env, '_reverse_out_steps', 0)
    if reverse_steps > 0:
        env._reverse_out_steps = reverse_steps - 1
        return getattr(env, '_reverse_out_dir', np.array([0.0, 0.0])), True

    # Door traversal mode
    if getattr(env, '_door_traversing', False):
        env._door_traversal_steps = getattr(env, '_door_traversal_steps', 0) + 1
        steps_in = env._door_traversal_steps
        pos = np.array(env.xy_coordinates[-1])

        gap_dir = getattr(env, '_door_gap_direction', None)
        if gap_dir is None:
            goal_angle = _goal_angle(env)
            gap_dir = np.array([np.cos(goal_angle), np.sin(goal_angle)])

        # Exit condition: no walls nearby = we're through the gap.
        if steps_in > 100 and not _walls_nearby(env, threshold=1.5):
            start = getattr(env, '_door_traversal_start', pos)
            dist_moved = np.linalg.norm(pos - start)
            pass
            env._door_traversing = False
            env._door_cooldown = 500
            return gap_dir, True

        # Stuck check (every 500 steps): narrow doors allow only slow crawl.
        if steps_in % 500 == 0 and steps_in > 0:
            check_pos = getattr(env, '_door_check_pos', pos)
            dist_recent = np.linalg.norm(pos - check_pos)
            env._door_check_pos = pos.copy()
            if dist_recent < 0.03:
                # Truly stuck — record this position in blocked-doors and reverse out.
                start = getattr(env, '_door_traversal_start', pos)
                blocked_doors = getattr(env, '_blocked_door_positions', [])
                blocked_doors.append(start.copy())
                env._blocked_door_positions = blocked_doors
                env._door_traversing = False
                env._door_cooldown = 500
                reverse_dir = -gap_dir
                env._reverse_out_steps = 500
                env._reverse_out_dir = reverse_dir
                return reverse_dir, True

        # Safety limit: 3000 steps max in any gap.
        if steps_in >= 3000:
            start = getattr(env, '_door_traversal_start', pos)
            dist_moved = np.linalg.norm(pos - start)
            env._door_traversing = False
            env._door_cooldown = 500
            pass
            return np.array([0.0, 0.0]), False

        return gap_dir, True

    if not env.wall_follow_active:

        if getattr(env, '_has_path', False):
            cur_hops = getattr(env, '_graph_dist_to_goal', None)
            if cur_hops is not None and cur_hops <= _NEAR_GOAL_HOPS:
                return np.array([0.0, 0.0]), False

        frustrated, goal_angle = _is_frustrated(env)
        if not frustrated or not _walls_nearby(env, threshold=1.5):
            return np.array([0.0, 0.0]), False

        # Frustrated with walls nearby → start wall following.
        pos = np.array(env.xy_coordinates[-1])
        goal_loc = np.array(env.goal_location)

        env.wall_follow_active = True
        env.wall_follow_direction = 1  # start normal (toward goal)
        env.wall_follow_steps = 0
        env.wall_follow_reversals = 0
        env._wall_follow_start_pos = pos.copy()

        pass

        return _compute_wall_follow_vector(env, goal_angle, nr_steps), True

    # Already wall-following
    goal_angle = _goal_angle(env)
    env.wall_follow_steps += 1

    # Door opening toward goal takes priority.
    gap_dir = _detect_door_opening(env, goal_angle)
    if gap_dir is not None:
        pos = env.xy_coordinates[-1]
        gap_angle_deg = np.degrees(np.arctan2(gap_dir[1], gap_dir[0]))
        pass
        env.wall_follow_active = False
        # Start behavior-based traversal — keep going until through or stuck.
        env._door_traversing = True
        env._door_traversal_steps = 0
        env._door_traversal_start = np.array(pos).copy()
        env._door_check_pos = np.array(pos).copy()
        env._last_door_pos = np.array(pos).copy()
        env._door_gap_direction = gap_dir.copy()
        return gap_dir, True

    # Deactivate if no walls nearby (reached open space).
    if not _walls_nearby(env, threshold=2.0):
        pos = env.xy_coordinates[-1]
        pass
        env.wall_follow_active = False
        return np.array([0.0, 0.0]), False

    # Stuck detection: if we haven't moved >0.5 m in the last 500 steps, reverse.
    if env.wall_follow_steps % 500 == 0 and env.wall_follow_steps > 0:
        pos = np.array(env.xy_coordinates[-1])
        start_pos = getattr(env, '_wall_follow_start_pos', pos)
        dist_moved = np.linalg.norm(pos - start_pos)
        if dist_moved < 0.5:
            env.wall_follow_direction *= -1
            env.wall_follow_reversals += 1
            dir_name = "toward-goal" if env.wall_follow_direction == 1 else "reversed"
            if env.wall_follow_reversals >= 3 and dist_moved < 0.1:
                env.retreat_to_safe_position()
                env.wall_follow_active = False
                env.wall_follow_reversals = 0
                env._door_cooldown = 1000
                return np.array([0.0, 0.0]), False
        else:
            env.wall_follow_reversals = 0
        env._wall_follow_start_pos = pos.copy()

    # Max time limit → reverse direction
    if env.wall_follow_steps >= env.wall_follow_max_steps:
        env.wall_follow_direction *= -1
        env.wall_follow_steps = 0
        env.wall_follow_reversals += 1
        pass

    return _compute_wall_follow_vector(env, goal_angle, nr_steps), True


def compute_navigation_goal_vector(gc_network, pc_network, cognitive_map, nr_steps, env,
                                   model="linear_lookahead", pod=None, spike_detector=None,
                                   a2c_explorer=None, spatial_grid=None, ai_explorer=None):
    """Computes the goal vector for the agent to travel to."""
    goal_pc = np.argmax(cognitive_map.reward_cells) if len(cognitive_map.reward_cells) > 0 else None
    current_pc = getattr(env, 'current_pc_idx', None)
    has_path = cognitive_map.path_exists(current_pc, goal_pc) if goal_pc is not None else False

    env._has_path = has_path

    # If a graph path exists, force topology-based navigation and disable
    # exploration-mode carry-over.
    if has_path:
        if env.exploration_mode:
            env.exploration_mode = False
            env.exploration_step_count = 0
            env.a2c_action_counter = 0
            env.a2c_current_action = None
            env.a2c_goal_vector = None
        env._zero_reward_lookahead_count = 0


    thigmotaxis_enabled = getattr(env, 'thigmotaxis_enabled', True)
    if thigmotaxis_enabled and not has_path:
        thigmo_vec, thigmo_active = _compute_thigmotaxis_vector(env, nr_steps)
        if thigmo_active:
            env.goal_vector = thigmo_vec
            env.goal_vector_original = thigmo_vec
            return

    use_a2c = (not has_path)

    if use_a2c and (a2c_explorer is not None or ai_explorer is not None) and goal_pc is not None:
        if not env.exploration_mode:
            pass
            env.exploration_mode = True
            env.exploration_step_count = 0
            env.a2c_action_counter = 0
            env._zero_reward_lookahead_count = 0

        env.exploration_step_count += 1

        # Periodically re-check if lookahead can find reward.
        path_recheck = env.path_recheck_interval
        if env.exploration_step_count % path_recheck == 0:
            can_path = cognitive_map.path_exists(current_pc, goal_pc)
            if can_path:
                pass
                env.exploration_mode = False
                env.a2c_current_action = None
                env.a2c_goal_vector = None
                env._zero_reward_lookahead_count = 0
                return

        if ai_explorer is not None:
            _compute_active_inference_goal_vector(env, ai_explorer, spatial_grid, nr_steps)
        else:
            _compute_a2c_goal_vector(env, a2c_explorer, spatial_grid, cognitive_map, current_pc, nr_steps)
        return


    if has_path:
        distance_to_goal = np.linalg.norm(env.goal_vector)
        distance_to_goal_original = np.linalg.norm(env.goal_vector_original)

        commit_remaining = max(0, getattr(env, '_topology_commit_counter', 0))
        if commit_remaining > 0:
            env._topology_commit_counter = commit_remaining - 1

        # Stuck-skip: if speed is near zero for many steps while topology says
        # we're still traveling toward the subgoal, advance to the next subgoal
        # since the current one is likely physically unreachable.
        recent_speed = (float(np.linalg.norm(env.xy_speeds[-1]))
                        if env.xy_speeds else 1.0)
        if recent_speed < 0.05:
            env._topo_stuck_steps = getattr(env, '_topo_stuck_steps', 0) + 1
        else:
            env._topo_stuck_steps = 0
        topo_stuck = getattr(env, '_topo_stuck_steps', 0) > 300

        update_fraction = 0.2 if model == "linear_lookahead" else 0.5
        topo_trigger = env.topology_based and distance_to_goal < 0.5
        if (topo_trigger or topo_stuck) and commit_remaining == 0:
            pick_intermediate_goal_vector(gc_network, pc_network, cognitive_map, env)
            env._topo_stuck_steps = 0
        elif (not env.topology_based and distance_to_goal / distance_to_goal_original < update_fraction
              and distance_to_goal_original > 0.3) or nr_steps == 0:
            find_new_goal_vector(gc_network, pc_network, cognitive_map, env,
                                 model=model, pod=pod, spike_detector=spike_detector)
        else:
            env.goal_vector = env.goal_vector - np.array(env.xy_speeds[-1]) * env.dt
            # While committed, floor goal-vector magnitude above the 0.5
            # recompute threshold so we don't re-enter the trigger every frame.
            # Direction is preserved; only magnitude is floored.
            if commit_remaining > 0 and topo_trigger:
                norm = np.linalg.norm(env.goal_vector)
                if 1e-3 < norm < 0.5:
                    env.goal_vector = env.goal_vector / norm * 0.5
                    env.goal_vector_original = env.goal_vector.copy()
        return

    # Fallback: no path and no A2C.
    distance_to_goal = np.linalg.norm(env.goal_vector)
    distance_to_goal_original = np.linalg.norm(env.goal_vector_original)

    update_fraction = 0.2 if model == "linear_lookahead" else 0.5
    if env.topology_based and (distance_to_goal < 0.3 or nr_steps == 0):
        pick_intermediate_goal_vector(gc_network, pc_network, cognitive_map, env)
    else:
        env.goal_vector = env.goal_vector - np.array(env.xy_speeds[-1]) * env.dt


def _compute_a2c_goal_vector(env, a2c_explorer, spatial_grid, cognitive_map, current_pc, nr_steps):
    """Compute goal vector using A2C exploration (learned policy + novelty gradient)."""
    # Pick a new A2C action if counter expired — recompute full blended vector.
    if env.a2c_action_counter <= 0:
        robot_pose = (env.xy_coordinates[-1][0], env.xy_coordinates[-1][1],
                      env.orientation_angle[-1] if env.orientation_angle else 0.0)

        state = build_exploration_state(robot_pose, cognitive_map, current_pc,
                                        arena_size=env.arena_size,
                                        goal_location=getattr(env, 'goal_location', None))
        action, log_prob, value = a2c_explorer.act(state)

        env.pending_transition = {
            'state': state,
            'action': action,
            'log_prob': log_prob,
            'value': value
        }

        env.a2c_current_action = action
        env.a2c_action_counter = env.a2c_action_repeat

        heading = env.orientation_angle[-1] if env.orientation_angle else 0.0
        a2c_vec = action_to_goal_vector(action, heading=heading, step_size=1.0)
        env.a2c_goal_vector = a2c_vec

        # Spatial novelty gradient (sampled once per action, not every step).
        novelty_vec = np.array([0.0, 0.0])
        if spatial_grid is not None:
            pos = env.xy_coordinates[-1]
            novelty_vec = spatial_grid.compute_novelty_direction(pos[0], pos[1])

        # Goal direction (distal landmark cues).
        goal_dir = np.array([0.0, 0.0])
        pos = env.xy_coordinates[-1] if env.xy_coordinates else None
        if pos is not None:
            goal_loc = env.goal_location if hasattr(env, 'goal_location') else None
            if goal_loc is not None:
                diff = np.array(goal_loc) - np.array(pos)
                d = np.linalg.norm(diff)
                if d > 0.1:
                    goal_dir = diff / d

        # Blend once and cache for the full repeat duration.
        goal_vec = 0.50 * a2c_vec + 0.45 * novelty_vec + 0.05 * goal_dir
        norm = np.linalg.norm(goal_vec)
        if norm > 0.01:
            goal_vec = goal_vec / norm

        env._cached_blend_vector = goal_vec

        pass

    env.a2c_action_counter -= 1

    goal_vec = getattr(env, '_cached_blend_vector', np.array([0.0, 0.0]))

    env.goal_vector = goal_vec
    env.goal_vector_original = goal_vec


def _compute_active_inference_goal_vector(env, ai_explorer, spatial_grid, nr_steps):
    """Compute goal vector using Active Inference."""
    if env.a2c_action_counter <= 0:
        pos = np.array(env.xy_coordinates[-1])
        goal_pos = np.array(env.goal_location)

        probe_angles = np.linspace(0, 2 * np.pi, ai_explorer.num_directions, endpoint=False)
        ray_distances = _get_ray_distances_toward(env, probe_angles)

        spatial_grid.visit(pos[0], pos[1], nr_steps)

        goal_vec = ai_explorer.select_action(pos, goal_pos, spatial_grid, ray_distances, nr_steps)

        norm = np.linalg.norm(goal_vec)
        if norm > 0.01:
            goal_vec = goal_vec / norm

        env._cached_blend_vector = goal_vec
        env.a2c_action_counter = ai_explorer.action_repeat

    env.a2c_action_counter -= 1

    goal_vec = getattr(env, '_cached_blend_vector', np.array([0.0, 0.0]))
    env.goal_vector = goal_vec
    env.goal_vector_original = goal_vec


def find_new_goal_vector(gc_network, pc_network, cognitive_map, env,
                         model="linear_lookahead", pod=None, spike_detector=None):
    """For vector-based navigation, computes goal vector with one grid-cell decoder."""

    video = False
    plot = False

    if model == "spike_detection":
        vec_avg_overall = spike_detector.compute_direction_signal(gc_network.gc_modules)
        env.goal_vector = vec_avg_overall
    elif model == "phase_offset_detector" and pod is not None:
        env.goal_vector = pod.compute_goal_vector(gc_network.gc_modules)
    else:
        env.goal_vector = perform_lookahead_directed(gc_network, pc_network,
                                                     cognitive_map, env)

    env.goal_vector_original = env.goal_vector


def pick_intermediate_goal_vector(gc_network, pc_network, cognitive_map, env, start_pc=None):
    """For topology-based navigation, compute sub goal vector with directed linear lookahead."""
    if getattr(env, '_topology_commit_counter', 0) > 0:
        return
    env.goal_vector = perform_lookahead_directed(gc_network, pc_network, cognitive_map, env)
    env.goal_vector_original = env.goal_vector
    # Commit window after a lookahead — perform_lookahead_directed is the most
    # expensive call in the loop. Two tiers:
    #   - High confidence (best >= 0.85): 200-frame commit.
    #   - Low confidence (best <  0.85): 10-frame commit so repeated lookaheads
    #     can still reassert the true gradient via running-average smoothing.
    if getattr(env, '_last_lookahead_best_reward', 0.0) >= 0.85:
        env._topology_commit_counter = 200
    else:
        env._topology_commit_counter = 10
