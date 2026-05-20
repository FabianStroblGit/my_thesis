import numpy as np


class RandomExplorer:

    def __init__(self, num_directions=16, action_repeat=200,
                 wall_margin=0.3, rng_seed=None):
        self.num_directions = num_directions
        self.action_repeat = action_repeat
        self.wall_margin = wall_margin
        self._rng = np.random.default_rng(rng_seed)

    def select_action(self, agent_pos, goal_pos, spatial_grid, ray_distances, current_step):
        """Sample a uniformly-random unblocked direction."""
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
