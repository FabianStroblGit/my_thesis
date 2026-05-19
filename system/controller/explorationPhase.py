import numpy as np


def compute_exploration_goal_vector(env, i):
    """Determines goal vector where the agent should head to"""
    position = env.xy_coordinates[-1]
    env.goal_vector = env.goal - position

    distance_to_goal = np.linalg.norm(env.goal_vector)
    # Check if agent has reached the (sub) goal
    if distance_to_goal < 0.2 or i == 0:
        if env.env_model == "linear_sunburst":
            navigate_to_location(env)  # Pick next goal to travel tp
            # pick_random_location(env, rectangular=True)
        elif env.env_model == "single_line_traversal":
            if i == 0:
                pick_random_straight_line(env)  # Pick random location at the beginning
                # print("Heading to ", env.goal)
        else:
            pick_random_location(env)


def pick_random_straight_line(env):
    """Picks a location at circular edge of environment"""
    angle = np.random.uniform(0, 2 * np.pi)
    env.goal = env.xy_coordinates[0] + np.array([np.cos(angle), np.sin(angle)]) * env.arena_size


def pick_random_location(env, rectangular=False):
    if not rectangular:
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0.5, 1) * env.arena_size
        env.goal = np.array([np.cos(angle), np.sin(angle)]) * distance
    else:
        x = np.random.uniform(0, 11)
        y = np.random.uniform(7, 11)
        env.goal = np.array([x, y])


def navigate_to_location(env):
    """Pre-coded exploration path for linear sunburst maze.

    When ``env.skip_left_lower_init`` is True, the leftward sweep of the
    lower corridor ([5.5,4.5] → [1.5,4.5]) is omitted so that no place cells
    are created at the three leftmost positions of the lower corridor.
    Used by the A2C benchmark INIT to control which PCs are present in the
    map before navigation starts.
    """
    if getattr(env, 'skip_left_lower_init', False):
        # Skip the leftward sweep — go straight east from the start so the
        # three leftmost lower-corridor PCs ([1.5,4.5], [2.5,4.5], [3.5,4.5])
        # are never created.
        goals = np.array([
                 [5.5, 4.5],
                 [9.5, 4.5],
                 [9.5, 7.5],
                 [10.5, 7.5],
                 [10.5, 10],
                 [8.5, 10],
                 [8.5, 7.5],
                 [6.5, 7.5],
                 [6.5, 10],
                 [4.5, 10],
                 [4.5, 7.5],
                 [2.5, 7.5],
                 [2.5, 10],
                 [0.5, 10],
                 [0.5, 7.5],
                 [2.5, 7.5],
                 [2.5, 10],
                 [4.5, 10],
                 [4.5, 7.5],
                 [6.5, 7.5],
                 [6.5, 10],
                 [8.5, 10],
                 [8.5, 7.5],
                 [9.5, 7.5]
                 ])
    else:
        goals = np.array([
                 [5.5, 4.5],
                 [1.5, 4.5],
                 [9.5, 4.5],
                 [9.5, 7.5],
                 [10.5, 7.5],
                 [10.5, 10],
                 [8.5, 10],
                 [8.5, 7.5],
                 [6.5, 7.5],
                 [6.5, 10],
                 [4.5, 10],
                 [4.5, 7.5],
                 [2.5, 7.5],
                 [2.5, 10],
                 [0.5, 10],
                 [0.5, 7.5],
                 [2.5, 7.5],
                 [2.5, 10],
                 [4.5, 10],
                 [4.5, 7.5],
                 [6.5, 7.5],
                 [6.5, 10],
                 [8.5, 10],
                 [8.5, 7.5],
                 [9.5, 7.5]
                 ])

    idx = env.goal_idx
    if idx < goals.shape[0]:
        env.goal = goals[idx]
        print("Heading to goal: ", idx, env.goal)
        env.goal_idx = idx + 1
