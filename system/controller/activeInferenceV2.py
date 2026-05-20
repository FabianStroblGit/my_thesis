
import numpy as np


class ActiveInferenceV2Explorer:
    """Multi-step AIF planner over the cognitive-map topology graph."""

    def __init__(self, cognitive_map, pc_network, gc_network=None, *,
                 horizon=3,
                 branching=4,
                 epistemic_weight=1.0,
                 extrinsic_weight=2.0,
                 inverse_temperature=2.0,
                 temporal_discount=0.9,
                 action_repeat=200,
                 momentum_weight=0.2,
                 wall_margin=0.5,
                 anisotropy_weight=1.5,
                 virtual_exploration=True,
                 virtual_radius=1.5,
                 virtual_min_real_dist=0.8):
        
        self.cognitive_map = cognitive_map
        self.pc_network = pc_network
        self.gc_network = gc_network
        self.horizon = int(horizon)
        self.branching = int(branching)
        self.epistemic_weight = float(epistemic_weight)
        self.extrinsic_weight = float(extrinsic_weight)
        self.inverse_temperature = float(inverse_temperature)
        self.temporal_discount = float(temporal_discount)
        self.action_repeat = int(action_repeat)
        self.momentum_weight = float(momentum_weight)
        self.wall_margin = float(wall_margin)
        self.anisotropy_weight = float(anisotropy_weight)
        self.virtual_exploration = bool(virtual_exploration)
        self.virtual_radius = float(virtual_radius)
        self.virtual_min_real_dist = float(virtual_min_real_dist)
        # navigationPhase reads num_directions to build the raycast probe-angle array.
        self.num_directions = 16
        # Local visit-count tally over PCs (drives the epistemic term).
        self._visit_counts = np.zeros(0, dtype=np.float32)
        self._prev_direction = None
        # Virtual PCs generated for the current select_action call. Indices
        # start at `len(pc_network.place_cells)` so they don't collide with
        # real PC indices. Cleared at the top of each select_action.
        self._virtual_pos = {}

    def _ensure_visit_counts(self):
        """Grow the visit-count array if the PC network has expanded."""
        n = len(self.pc_network.place_cells)
        if n > len(self._visit_counts):
            grown = np.zeros(n, dtype=np.float32)
            grown[: len(self._visit_counts)] = self._visit_counts
            self._visit_counts = grown

    def _belief(self, agent_pos):
        """Return (most-likely PC index, normalised belief Q(s)) over PCs.

        With a grid-cell network, Q(s) is the softmax of PC firing values;
        otherwise Q(s) is a Gaussian over Euclidean distance from the
        agent's position to each PC's env_coordinates.
        """
        n = len(self.pc_network.place_cells)
        if n == 0:
            return None, np.zeros(0, dtype=np.float32)

        if self.gc_network is not None:
            firing = self.pc_network.compute_firing_values(self.gc_network.gc_modules)
            firing = np.asarray(firing, dtype=np.float32)
            if firing.max() <= 0:
                return self._euclidean_belief(agent_pos)
            Q = firing / firing.sum()
            return int(np.argmax(Q)), Q
        return self._euclidean_belief(agent_pos)

    def _euclidean_belief(self, agent_pos):
        coords = np.array(
            [pc.env_coordinates for pc in self.pc_network.place_cells],
            dtype=np.float32,
        )
        d2 = np.sum((coords - np.asarray(agent_pos, dtype=np.float32)) ** 2, axis=1)
        scores = -d2
        scores -= scores.max()
        Q = np.exp(scores)
        Q /= Q.sum()
        return int(np.argmax(Q)), Q

    def _is_virtual(self, pc_idx):
        """True if pc_idx refers to a virtual PC stand-in."""
        return pc_idx in self._virtual_pos

    def _coord_of(self, pc_idx):
        """Coordinates of a real or virtual PC, or None when unavailable."""
        if pc_idx in self._virtual_pos:
            return self._virtual_pos[pc_idx]
        if pc_idx < len(self.pc_network.place_cells):
            coord = self.pc_network.place_cells[pc_idx].env_coordinates
            return None if coord is None else np.asarray(coord, dtype=np.float32)
        return None

    def _visit_count_of(self, pc_idx):
        """N(s) for the epistemic term. Virtual PCs are N=0 by construction."""
        if pc_idx in self._virtual_pos:
            return 0.0
        if pc_idx < len(self._visit_counts):
            return float(self._visit_counts[pc_idx])
        return 0.0

    def _reward_of(self, pc_idx):
        """Extrinsic reward at the target. Virtual PCs have r=0 since their
        reward field is unknown; their pull comes from epistemic + anisotropic
        terms only."""
        if pc_idx in self._virtual_pos:
            return 0.0
        if pc_idx < len(self.cognitive_map.reward_cells):
            return float(self.cognitive_map.reward_cells[pc_idx])
        return 0.0

    def _generate_virtual_candidates(self, agent_arr, goal_dir, ray_distances,
                                     real_neighbour_ids):
        """Create virtual PC stand-ins in goal-direction compass slots that
        are wall-clear and not already covered by a real neighbour."""
        self._virtual_pos.clear()
        if not self.virtual_exploration or ray_distances is None:
            return []
        n_rays = len(ray_distances)
        if n_rays == 0:
            return []

        # Coordinates of existing real-neighbour PCs (for duplicate suppression).
        real_coords = []
        for nb in real_neighbour_ids:
            c = self._coord_of(nb)
            if c is not None:
                real_coords.append(c)

        base_idx = len(self.pc_network.place_cells)
        angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        out = []
        # Minimum forward clearance required at a virtual's compass angle.
        clear_required = self.wall_margin + self.virtual_radius * 0.5
        for i, ang in enumerate(angles):
            if ray_distances[i] < clear_required:
                continue
            dir_vec = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
            # Goal-half-plane filter loosened to ~215 deg wedge; -0.3 still
            # excludes directly-backward virtuals.
            if goal_dir is not None and float(np.dot(dir_vec, goal_dir)) < -0.3:
                continue
            candidate = agent_arr + self.virtual_radius * dir_vec
            # Suppress if a real neighbour is already nearby.
            too_close = False
            for c in real_coords:
                if float(np.linalg.norm(candidate - c)) < self.virtual_min_real_dist:
                    too_close = True
                    break
            if too_close:
                continue
            idx = base_idx + len(out)
            self._virtual_pos[idx] = candidate
            out.append(idx)
        return out

    def _neighbours(self, pc_idx, top_k):
        """Top-k neighbouring PC indices of `pc_idx` along topology edges.

        When the node has more than `top_k` neighbours, prefer higher-reward
        ones. Virtual PC indices have no topology edges and return [].
        """
        if pc_idx is None:
            return []
        if pc_idx >= self.cognitive_map.topology_cells.shape[0]:
            return []
        adjacency = self.cognitive_map.topology_cells[pc_idx]
        neighbours = np.where(adjacency == 1)[0]
        if len(neighbours) == 0:
            return []
        if len(neighbours) <= top_k:
            return neighbours.tolist()
        rewards = self.cognitive_map.reward_cells[neighbours]
        order = np.argsort(-rewards)[:top_k]
        return neighbours[order].tolist()

    def _reachable_neighbours(self, pc_idx, agent_pos, ray_distances, top_k):
        """First-hop neighbours whose direction from the agent is unobstructed.

        Like `_neighbours` but drops candidates whose direction from the
        agent's current position is blocked by a wall (raycast in that
        direction < ``self.wall_margin``). Used only for the first hop.
        """
        cands = self._neighbours(pc_idx, top_k)
        if ray_distances is None or len(ray_distances) == 0 or agent_pos is None:
            return cands
        n_rays = len(ray_distances)
        agent = np.asarray(agent_pos, dtype=np.float32)
        keep = []
        for nb in cands:
            coord = self.pc_network.place_cells[nb].env_coordinates
            if coord is None:
                keep.append(nb)
                continue
            disp = np.asarray(coord, dtype=np.float32) - agent
            norm = float(np.linalg.norm(disp))
            if norm < 1e-3:
                keep.append(nb)
                continue
            ang = float(np.arctan2(disp[1], disp[0])) % (2 * np.pi)
            idx = int(np.round(ang / (2 * np.pi) * n_rays)) % n_rays
            if ray_distances[idx] >= self.wall_margin:
                keep.append(nb)
        return keep

    def _enumerate_policies(self, start_pc, agent_pos=None, ray_distances=None,
                            goal_dir=None):
        """Build every policy of length up to `horizon` starting at start_pc.

        Partial policies that terminate at dead ends are kept. The first hop
        is filtered by `_reachable_neighbours` and augmented with virtual PC
        stand-ins when ``self.virtual_exploration`` is True. Deeper hops use
        the unfiltered topology neighbour set since ray data doesn't apply
        to imagined future positions.
        """
        if start_pc is None:
            return []
        frontier = [(int(start_pc), [])]
        all_policies = []
        for h in range(self.horizon):
            next_frontier = []
            for cur_pc, history in frontier:
                if h == 0 and ray_distances is not None:
                    real = self._reachable_neighbours(
                        cur_pc, agent_pos, ray_distances, self.branching
                    )
                    agent_arr = (np.asarray(agent_pos, dtype=np.float32)
                                 if agent_pos is not None else None)
                    if agent_arr is not None:
                        virtuals = self._generate_virtual_candidates(
                            agent_arr, goal_dir, ray_distances, real
                        )
                    else:
                        virtuals = []
                    neighbours = real + virtuals
                else:
                    neighbours = self._neighbours(cur_pc, self.branching)
                if not neighbours:
                    if history:
                        all_policies.append(history)
                    continue
                for nb in neighbours:
                    next_frontier.append((int(nb), history + [int(nb)]))
            frontier = next_frontier
            if not frontier:
                break
        for _cur_pc, history in frontier:
            if history:
                all_policies.append(history)
        return all_policies

    def _goal_heading(self, agent_pos, goal_pos):
        """Unit vector from agent toward goal; None if undefined."""
        if goal_pos is None or agent_pos is None:
            return None
        d = np.asarray(goal_pos, dtype=np.float32) - np.asarray(agent_pos, dtype=np.float32)
        n = float(np.linalg.norm(d))
        if n < 1e-3:
            return None
        return d / n

    def _anisotropic_gain(self, pc_idx, agent_arr, goal_dir):
        """Multiplicative boost on the epistemic term for goal-aligned PCs.

        Gain = 1 + alpha * max(0, cos theta), where theta is the angle
        between agent->PC displacement and agent->goal direction.
        """
        if goal_dir is None or self.anisotropy_weight <= 0.0:
            return 1.0
        coord = self._coord_of(pc_idx)
        if coord is None:
            return 1.0
        disp = coord - agent_arr
        n = float(np.linalg.norm(disp))
        if n < 1e-3:
            return 1.0
        cos_sim = float(np.dot(disp / n, goal_dir))
        return 1.0 + self.anisotropy_weight * max(0.0, cos_sim)

    def _expected_free_energy(self, policy, agent_arr, goal_dir):
        """G(pi) for a single policy = sequence of target PC indices.

        Lower G is preferred. The bracketed sum is negated so that
        minimising G maximises the discounted (epistemic + extrinsic) value.
        """
        total = 0.0
        for tau, pc_idx in enumerate(policy, start=1):
            gamma = self.temporal_discount ** (tau - 1)
            v = self._visit_count_of(pc_idx)
            epistemic = np.log(1.0 + 1.0 / (1.0 + v))
            gain = self._anisotropic_gain(pc_idx, agent_arr, goal_dir)
            extrinsic = self._reward_of(pc_idx)
            total += gamma * (self.epistemic_weight * gain * epistemic + self.extrinsic_weight * extrinsic)
        return -total


    def select_action(self, agent_pos, goal_pos, spatial_grid, ray_distances, current_step):
        """Pick the next direction via AIF policy enumeration + softmax."""
        self._ensure_visit_counts()
        # Clear virtual PC stand-ins; regenerated per call since the
        # agent's position, wall geometry, and goal direction all change.
        self._virtual_pos = {}

        start_pc, _Q = self._belief(agent_pos)
        if start_pc is None:
            return np.array([0.0, 0.0])

        agent_arr = np.asarray(agent_pos, dtype=np.float32)
        goal_dir = self._goal_heading(agent_pos, goal_pos)

        policies = self._enumerate_policies(
            start_pc, agent_pos=agent_pos, ray_distances=ray_distances,
            goal_dir=goal_dir,
        )
        if not policies:
            # All first-hop neighbours wall-blocked. Fall back to unfiltered
            # enumeration so the AIF marginal can surface a least-bad direction.
            policies = self._enumerate_policies(start_pc)
        if not policies:
            return np.array([0.0, 0.0])

        G_values = np.array(
            [self._expected_free_energy(p, agent_arr, goal_dir) for p in policies],
            dtype=np.float32,
        )

        # Softmax over policies: P(pi) ~ exp(-tau^-1 G(pi)).
        scores = -self.inverse_temperature * G_values
        scores -= scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()

        # Marginalise to first-step action.
        first_action_weights = {}
        for w, p in zip(weights, policies):
            a = p[0]
            first_action_weights[a] = first_action_weights.get(a, 0.0) + float(w)

        # Optional head-direction momentum bonus on the marginal (not in G).
        if self._prev_direction is not None and self.momentum_weight > 0.0:
            for a, w in list(first_action_weights.items()):
                target = self._coord_of(a)
                if target is None:
                    continue
                direction = target - agent_arr
                norm = np.linalg.norm(direction)
                if norm < 1e-3:
                    continue
                cos_sim = float(np.dot(direction / norm, self._prev_direction))
                first_action_weights[a] = w + self.momentum_weight * cos_sim

        best_action, _best_w = max(first_action_weights.items(), key=lambda kv: kv[1])

        target_coord = self._coord_of(best_action)
        if target_coord is None:
            return np.array([0.0, 0.0])
        direction = target_coord - agent_arr
        norm = float(np.linalg.norm(direction))
        if norm < 1e-3:
            return np.array([0.0, 0.0])
        vec = direction / norm

        # Respect the live raycast block-out filter.
        if ray_distances is not None and len(ray_distances) > 0:
            n_rays = len(ray_distances)
            angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
            chosen_angle = float(np.arctan2(vec[1], vec[0])) % (2 * np.pi)
            idx = int(np.round(chosen_angle / (2 * np.pi) * n_rays)) % n_rays
            if ray_distances[idx] < self.wall_margin:
                return np.array([0.0, 0.0])

        # Reinforce visit count of the chosen target.
        if best_action < len(self._visit_counts):
            self._visit_counts[best_action] += 1.0

        self._prev_direction = vec

        if current_step % 500 == 0:
            target_kind = "virt" if self._is_virtual(best_action) else "real"
            n_virtual = len(self._virtual_pos)
            print(f"[AI2] Step {current_step}: pos=({agent_pos[0]:.1f},{agent_pos[1]:.1f}), "
                  f"current_pc={start_pc}, target_pc={best_action}({target_kind}), "
                  f"P(a*)={first_action_weights[best_action]:.3f}, "
                  f"|policies|={len(policies)}, virtuals={n_virtual}")

        return vec
