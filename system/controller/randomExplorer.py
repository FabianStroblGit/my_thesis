"""
randomExplorer.py — Uniform-random baseline explorer for the no-path branch.

Usage in config.json:
    "exploration": { "type": "random" }

Then run any benchmark with --random selected (see benchmark_random.py).
"""

import numpy as np


class RandomExplorer:

    def __init__(self, num_directions=16, action_repeat=200,
                 wall_margin=0.3, rng_seed=None):
        """
        Args:
            num_directions: candidate-direction count (matches raycast count).
            action_repeat: steps to commit to a chosen direction before
                resampling. Matches AIF default so the baseline operates at
                the same temporal resolution.
            wall_margin: metres of clearance required ahead of the agent for
                a direction to be considered "non-blocked". Same heuristic
                AIF uses (max_dist < 0.2 in its evaluator).
            rng_seed: optional integer for reproducibility.
        """
        self.num_directions = num_directions
        self.action_repeat = action_repeat
        self.wall_margin = wall_margin
        self._rng = np.random.default_rng(rng_seed)

    def select_action(self, agent_pos, goal_pos, spatial_grid, ray_distances, current_step):
        """Sample a uniformly-random unblocked direction.

        Signature matches ActiveInferenceV2Explorer.select_action so it can be
        plugged into _compute_active_inference_goal_vector in navigationPhase.

        Args:
            agent_pos: np.array [x, y]; ignored here (kept for API parity).
            goal_pos: np.array [x, y]; ignored.
            spatial_grid: SpatialExplorationGrid; ignored.
            ray_distances: np.ndarray of shape (num_directions,); used to
                filter out blocked directions.
            current_step: int, for logging.

        Returns:
            np.ndarray: unit vector for the chosen direction, or a zero
            vector if every direction is blocked.
        """
        angles = np.linspace(0, 2 * np.pi, self.num_directions, endpoint=False)


        unblocked = ray_distances > (self.wall_margin + 0.2)
        unblocked_idxs = np.where(unblocked)[0]

        if len(unblocked_idxs) == 0:
            return np.array([0.0, 0.0])

        choice = int(self._rng.choice(unblocked_idxs))
        angle = angles[choice]
        vec = np.array([np.cos(angle), np.sin(angle)])

        if current_step % 500 == 0:
            print(f"[RND] Step {current_step}: pos=({agent_pos[0]:.1f},{agent_pos[1]:.1f}), "
                  f"chose={np.degrees(angle):.0f}°  "
                  f"({len(unblocked_idxs)}/{self.num_directions} dirs unblocked)")

        return vec
