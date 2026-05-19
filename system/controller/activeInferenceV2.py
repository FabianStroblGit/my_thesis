
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
        # navigationPhase reads num_directions to build the raycast
        # probe-angle array passed into select_action().
        self.num_directions = 16
        # Local visit-count tally over PCs (drives the epistemic term). This
        # is the controller's own ledger; it grows as the agent commits to
        # target PCs during evaluation and is independent of the cognitive
        # map's recency cells.
        self._visit_counts = np.zeros(0, dtype=np.float32)
        self._prev_direction = None
        # Virtual PCs generated for the current select_action call. Indices
        # start at `len(pc_network.place_cells)` so they don't collide with
        # real PC indices. Cleared at the top of each select_action.
        self._virtual_pos = {}

    # ---- internals --------------------------------------------------------

    def _ensure_visit_counts(self):
        """Grow the visit-count array if the PC network has expanded."""
        n = len(self.pc_network.place_cells)
        if n > len(self._visit_counts):
            grown = np.zeros(n, dtype=np.float32)
            grown[: len(self._visit_counts)] = self._visit_counts
            self._visit_counts = grown

    def _belief(self, agent_pos):
        """Return (most-likely PC index, normalised belief Q(s)) over PCs.

        When a grid-cell network is available, Q(s) is the softmax of the
        PC firing values. Otherwise, Q(s) is a Gaussian over Euclidean
        distance from the agent's current position to each PC's
        env_coordinates.
        """
        n = len(self.pc_network.place_cells)
        if n == 0:
            return None, np.zeros(0, dtype=np.float32)

        if self.gc_network is not None:
            firing = self.pc_network.compute_firing_values(self.gc_network.gc_modules)
            firing = np.asarray(firing, dtype=np.float32)
            if firing.max() <= 0:
                # No PC firing — fall back to Euclidean.
                return self._euclidean_belief(agent_pos)
            # Normalise to a proper probability distribution.
            Q = firing / firing.sum()
            return int(np.argmax(Q)), Q
        return self._euclidean_belief(agent_pos)

    def _euclidean_belief(self, agent_pos):
        coords = np.array(
            [pc.env_coordinates for pc in self.pc_network.place_cells],
            dtype=np.float32,
        )
        d2 = np.sum((coords - np.asarray(agent_pos, dtype=np.float32)) ** 2, axis=1)
        # Softmax of -d² with a unit temperature gives a Gaussian-style belief.
        scores = -d2
        scores -= scores.max()
        Q = np.exp(scores)
        Q /= Q.sum()
        return int(np.argmax(Q)), Q

    # ---- real/virtual helpers --------------------------------------------

    def _is_virtual(self, pc_idx):
        """True if pc_idx refers to a virtual PC stand-in created during
        the current select_action call."""
        return pc_idx in self._virtual_pos

    def _coord_of(self, pc_idx):
        """Coordinates of a real or virtual PC, or None when unavailable.
        Virtual PCs live in ``self._virtual_pos``; real PCs in
        ``pc_network.place_cells``."""
        if pc_idx in self._virtual_pos:
            return self._virtual_pos[pc_idx]
        if pc_idx < len(self.pc_network.place_cells):
            coord = self.pc_network.place_cells[pc_idx].env_coordinates
            return None if coord is None else np.asarray(coord, dtype=np.float32)
        return None

    def _visit_count_of(self, pc_idx):
        """N(s) for the epistemic term. Virtual PCs are unvisited by
        construction (N=0), maximising their information-gain contribution.
        """
        if pc_idx in self._virtual_pos:
            return 0.0
        if pc_idx < len(self._visit_counts):
            return float(self._visit_counts[pc_idx])
        return 0.0

    def _reward_of(self, pc_idx):
        """Extrinsic reward at the target. Virtual PCs are imagined in
        regions where no real PC has been created, so the reward field is
        unknown — we set it to zero. The agent's pull toward goal-direction
        virtuals therefore comes entirely from the epistemic term plus the
        anisotropic gain, not from the (currently uninformative) reward
        gradient. This is theoretically defensible: imagined hidden states
        in unmapped space have an uncertain extrinsic value, which in AIF
        translates to a neutral preference contribution."""
        if pc_idx in self._virtual_pos:
            return 0.0
        if pc_idx < len(self.cognitive_map.reward_cells):
            return float(self.cognitive_map.reward_cells[pc_idx])
        return 0.0

    def _generate_virtual_candidates(self, agent_arr, goal_dir, ray_distances,
                                     real_neighbour_ids):
        """Create virtual PC stand-ins in goal-direction compass slots that
        are wall-clear and not already covered by a real neighbour.

        Virtual PCs let policy enumeration reach into unmapped space without
        leaving the AIF framework: each virtual is an imagined hidden state
        with N=0 and r=0, and policies starting at one are dead-end
        single-step plans (the topology graph has no edges from virtual
        indices, so deeper hops naturally terminate). The anisotropic gain
        and the wall-clear / no-real-cover filters together ensure that
        virtuals are generated exactly where the agent has reason to look:
        toward the goal, into open space, into a gap in the cognitive map.
        """
        self._virtual_pos.clear()
        if not self.virtual_exploration or ray_distances is None:
            return []
        n_rays = len(ray_distances)
        if n_rays == 0:
            return []

        # Coordinates of existing real-neighbour PCs (used to suppress
        # virtual candidates that would duplicate an already-real target).
        real_coords = []
        for nb in real_neighbour_ids:
            c = self._coord_of(nb)
            if c is not None:
                real_coords.append(c)

        base_idx = len(self.pc_network.place_cells)
        angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        out = []
        # Minimum forward clearance required at a virtual's compass angle:
        # leave self.wall_margin between the agent and any wall, plus
        # enough room to actually reach the virtual position. Without this
        # we'd place virtuals on the far side of nearby walls.
        clear_required = self.wall_margin + self.virtual_radius * 0.5
        for i, ang in enumerate(angles):
            if ray_distances[i] < clear_required:
                continue
            dir_vec = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
            # Goal-half-plane filter, loosened to a ~215° wedge so V2 can
            # also place virtuals at WSW/ESE angles. With a strict 180°
            # cone (dot >= 0) the agent gets pulled NW toward the closed
            # door every time, never sideways into the unmapped west
            # corridor. The −0.3 threshold still excludes directly-
            # backward virtuals (no information gain there).
            if goal_dir is not None and float(np.dot(dir_vec, goal_dir)) < -0.3:
                continue
            candidate = agent_arr + self.virtual_radius * dir_vec
            # Suppress if a real neighbour is already nearby; we want
            # virtuals to fill gaps in the topology graph, not duplicate
            # existing real candidates.
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

        When the node has more than `top_k` neighbours, prefer those with
        higher reward (closer to the goal in topology). This caps the
        branching factor in policy enumeration. Virtual PC indices have
        no topology edges by construction — they return an empty list,
        which terminates the policy expansion at h=0 (the virtual itself
        becomes a length-1 dead-end policy that still scores normally).
        """
        if pc_idx is None:
            return []
        # Virtual PCs have no topology edges; expansions from them stop here.
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

        Identical to `_neighbours` but additionally drops any candidate whose
        direction from the agent's current position is blocked by a wall
        (raycast in that direction < ``self.wall_margin``). Used only for the
        first hop of policy enumeration: subsequent hops are imagined future
        positions where ray data is not available, so they keep using
        `_neighbours` directly.

        Routing wall information into the policy enumeration means the EFE
        sum is computed only over reachable first-step targets. Without this,
        the planner repeatedly picks a goal-direction neighbour that the
        post-decision wall filter then silently rejects, returning the zero
        vector and pinning the agent against the wall. With this filter the
        planner sees only reachable neighbours and naturally picks south/east
        targets when north is blocked.
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
                # Without coordinates we can't check; keep it conservatively.
                keep.append(nb)
                continue
            disp = np.asarray(coord, dtype=np.float32) - agent
            norm = float(np.linalg.norm(disp))
            if norm < 1e-3:
                # Self-loop / coincident PC; not blocked by any wall.
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

        A policy is a list of target PC indices [s_1, …, s_H]. We also keep
        partial policies that terminate early at dead ends, so the agent can
        still consider stepping into a sink.

        When ``agent_pos`` and ``ray_distances`` are supplied, the first hop
        is restricted to neighbours whose direction from the agent is
        unobstructed (see `_reachable_neighbours`). Deeper hops keep using
        the unfiltered topology neighbour set because they correspond to
        imagined future positions where current ray data does not apply.

        When ``self.virtual_exploration`` is True and ``goal_dir`` is given,
        the first-hop candidate set is augmented with virtual PC
        stand-ins generated in wall-clear compass slots within the goal
        half-plane that no real neighbour covers. Virtual candidates have
        no topology edges, so policies starting with a virtual terminate
        as length-1 plans — the EFE machinery scores them like any other
        policy, with the anisotropic gain doing the work of preferring
        goal-aligned virtuals.
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
                    # Dead-end node: keep the partial policy if non-empty.
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

        Gain = 1 + α · max(0, cos θ), where θ is the angle between the
        agent→PC displacement and the agent→goal direction. PCs in the goal
        half-plane get an inflated novelty bonus; PCs behind/orthogonal keep
        their baseline novelty (gain = 1). Applies uniformly to real and
        virtual PCs (the coord lookup goes through ``_coord_of``).
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
        """G(π) for a single policy = sequence of target PC indices.

        Lower G is preferred. We negate the bracketed sum so that minimising
        G corresponds to maximising the discounted sum of (epistemic +
        extrinsic) value. The epistemic term is scaled by an anisotropic
        gain that favours PCs lying in the goal direction.

        Real and virtual PCs share the same scoring rule; the per-step
        N, r, and coordinate lookups go through ``_visit_count_of``,
        ``_reward_of``, and ``_coord_of`` so that virtual PCs (N=0, r=0)
        score by epistemic value + anisotropic gain alone.
        """
        total = 0.0
        for tau, pc_idx in enumerate(policy, start=1):
            gamma = self.temporal_discount ** (tau - 1)
            v = self._visit_count_of(pc_idx)
            epistemic = np.log(1.0 + 1.0 / (1.0 + v))
            gain = self._anisotropic_gain(pc_idx, agent_arr, goal_dir)
            extrinsic = self._reward_of(pc_idx)
            total += gamma * (self.epistemic_weight * gain * epistemic + self.extrinsic_weight * extrinsic)
        return -total  # AIF: minimise G; here G = -(value)


    def select_action(self, agent_pos, goal_pos, spatial_grid, ray_distances, current_step):
        """Pick the next direction via AIF policy enumeration + softmax."""
        self._ensure_visit_counts()
        # Clear virtual PC stand-ins from the previous call. They are
        # regenerated per call in _enumerate_policies because the
        # agent's position, wall geometry, and goal direction all change.
        self._virtual_pos = {}

        start_pc, _Q = self._belief(agent_pos)
        if start_pc is None:
            return np.array([0.0, 0.0])

        agent_arr = np.asarray(agent_pos, dtype=np.float32)
        goal_dir = self._goal_heading(agent_pos, goal_pos)

        # Pass current position + ray distances + goal direction into the
        # enumeration so the first hop is filtered against the wall-margin
        # raycast AND augmented with virtual goal-direction stand-ins when
        # the topology graph offers no goal-aligned reachable neighbour.
        policies = self._enumerate_policies(
            start_pc, agent_pos=agent_pos, ray_distances=ray_distances,
            goal_dir=goal_dir,
        )
        if not policies:
            # All first-hop neighbours are wall-blocked. Fall back to the
            # unfiltered enumeration so the planner can at least pick an
            # action — the post-decision filter will still suppress motion,
            # but the AIF marginal will surface the least-bad direction (e.g.
            # parallel to the wall) for retreat to act on.
            policies = self._enumerate_policies(start_pc)
        if not policies:
            return np.array([0.0, 0.0])

        G_values = np.array(
            [self._expected_free_energy(p, agent_arr, goal_dir) for p in policies],
            dtype=np.float32,
        )

        # Softmax over policies: P(π) ∝ exp(-τ⁻¹ G(π)).
        scores = -self.inverse_temperature * G_values
        scores -= scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()

        # Marginalise to first-step action.
        first_action_weights = {}
        for w, p in zip(weights, policies):
            a = p[0]
            first_action_weights[a] = first_action_weights.get(a, 0.0) + float(w)

        # Optional head-direction momentum bonus on the marginal, not in G.
        # Uses ``_coord_of`` so virtual PC stand-ins participate on equal
        # footing with real PCs.
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

        # Convert "move toward PC best_action" → unit direction vector.
        # ``_coord_of`` resolves both real PCs and virtual PC stand-ins.
        target_coord = self._coord_of(best_action)
        if target_coord is None:
            return np.array([0.0, 0.0])
        direction = target_coord - agent_arr
        norm = float(np.linalg.norm(direction))
        if norm < 1e-3:
            return np.array([0.0, 0.0])
        vec = direction / norm

        # Respect the live raycast block-out filter: if the chosen direction
        # is into a wall, return zero and let the caller handle it.
        if ray_distances is not None and len(ray_distances) > 0:
            n_rays = len(ray_distances)
            angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
            chosen_angle = float(np.arctan2(vec[1], vec[0])) % (2 * np.pi)
            idx = int(np.round(chosen_angle / (2 * np.pi) * n_rays)) % n_rays
            if ray_distances[idx] < self.wall_margin:
                return np.array([0.0, 0.0])

        # Reinforce the visit count of the chosen target so future planning
        # treats it as less novel.
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
