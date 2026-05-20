import numpy as np
import os
from collections import deque

# Concept based on Erdem 2012. For detailed explanations please refer to thesis or paper.

# Decisions
# Only one place cell is active at the same time -> winner takes it all
# Binary connections in cognitive map between two topology cells


class CognitiveMapNetwork:
    """The CognitiveMapNetwork keeps track of all recency, topology and reward cells"""
    def __init__(self, dt, from_data=False):
        if not from_data:
            self.recency_cells = np.array([])  # array of firing values between 0 and 1; 1 where the agent is
            self.topology_cells = np.zeros((1, 1))  # matrix of connections, size (#pc x #pc)
            self.reward_cells = np.array([])  # array of firing values between 0 and 1; 1 where the goal is
        else:
            self.topology_cells = np.load("data/cognitive_map/topology_cells.npy")
            self.reward_cells = np.load("data/cognitive_map/reward_cells.npy")
            self.recency_cells = np.zeros_like(self.reward_cells)
            self.recency_cells[0] = 1

        self.dt = dt

        # Visit tracking for A2C exploration intrinsic reward
        nr_pcs = len(self.recency_cells)
        self.visit_counts = np.zeros(nr_pcs, dtype=np.int32)
        self.last_visit_step = np.full(nr_pcs, -1, dtype=np.int32)

        epsilon = 2  # parameters tuned for velocity and environment size
        lam = 0.01 / dt  # parameters tuned for velocity and environment size
        self.decay_rate = epsilon**(-lam * dt)  # determines how quickly recency cell firing decays
        self.recency_threshold = 0.5  # determines when we still consider an recency cell as active

        # Index of the PC that was active in the previous time step.
        self.prior_idx_pc_firing = None

        self.active_threshold = 0.85  # determines when we consider a place cell to be active

    @property
    def nr_place_cells(self):
        """Return the number of place cells in the cognitive map."""
        return len(self.recency_cells)

    def add_cortical_column(self, reward):
        """Adds a set of three prefrontal cortex cells to network. Called when a new place cell was created."""

        # Add a recency cell to the end, currently active
        self.recency_cells = np.append(self.recency_cells, 1)

        # Extend topology cell array by a row and column, no connections have formed yet
        n = len(self.recency_cells)
        reference_array = np.zeros((n, n))
        reference_array[:self.topology_cells.shape[0], :self.topology_cells.shape[1]] = self.topology_cells
        self.topology_cells = reference_array

        # Add a reward cell to the end, reward value depends on if an reward has been found
        self.reward_cells = np.append(self.reward_cells, reward)

        self.visit_counts = np.append(self.visit_counts, 0)
        self.last_visit_step = np.append(self.last_visit_step, -1)

    def compute_reward_spiking(self, pc_firing):
        """Determine which place cells are active and multiply with reward value"""
        pc_firing = np.where(np.array(pc_firing) > self.active_threshold, pc_firing, 0)  # Check for active pc
        rewards = self.reward_cells * pc_firing  # Multiply with reward spiking
        idx_pc_active = np.argmax(rewards)
        reward = np.max(rewards)
        return [reward, idx_pc_active]  # Return highest reward and idx of pc

    def track_movement(self, pc_firing, created_new_pc, reward, env=None, current_step=0,
                        pc_network=None, max_connection_dist=1.5):
        """Keeps track of current place cell firing and creation of new place cells."""

        if created_new_pc:
            self.add_cortical_column(reward)

        idx_pc_active = np.argmax(pc_firing)  # max one place cell is considered as active
        pc_active = np.max(pc_firing)

        # Decay recency cells and refresh the active one.
        self.recency_cells = self.recency_cells * self.decay_rate
        if pc_active > self.active_threshold:
            self.recency_cells[idx_pc_active] = 1
            if env is not None:
                env.current_pc_idx = idx_pc_active
            if idx_pc_active < len(self.visit_counts):
                self.visit_counts[idx_pc_active] += 1
                self.last_visit_step[idx_pc_active] = current_step

        # Did we enter a different PC than last step?
        prior_pc = self.prior_idx_pc_firing
        if created_new_pc:
            entered_different_pc = True
        elif pc_active > self.active_threshold and prior_pc is not None and prior_pc != idx_pc_active:
            entered_different_pc = True
        elif pc_active > self.active_threshold and prior_pc is None:
            # First valid PC observation - record it but don't create connections yet.
            entered_different_pc = True
        else:
            entered_different_pc = False

        # Only create topology connections if a valid prior PC exists.
        if entered_different_pc and prior_pc is not None:
            # Ensure the prior PC has recency above threshold so a connection
            # can form (handles PCs loaded from data with recency initialised to 0).
            if prior_pc < len(self.recency_cells):
                if self.recency_cells[prior_pc] < self.recency_threshold:
                    self.recency_cells[prior_pc] = self.recency_threshold + 0.1

            prior_visited = np.heaviside(self.recency_cells - self.recency_threshold, 1)
            currently_visited = np.heaviside(self.recency_cells - 1, 1)
            new_connections = np.outer(prior_visited, currently_visited)

            # Filter out connections between PCs that are too far apart
            # (prevents false shortcuts through walls).
            if pc_network is not None:
                n = new_connections.shape[0]
                for i in range(n):
                    if not prior_visited[i]:
                        continue
                    pos_i = pc_network.place_cells[i].env_coordinates
                    if pos_i is None:
                        continue
                    for j in range(n):
                        if new_connections[i, j] == 0:
                            continue
                        pos_j = pc_network.place_cells[j].env_coordinates
                        if pos_j is None:
                            continue
                        if np.linalg.norm(pos_i - pos_j) > max_connection_dist:
                            new_connections[i, j] = 0

            # Save bilateral connections; skip reward propagation if no new edge was added.
            prev_edge_count = int(self.topology_cells.sum())
            self.topology_cells = np.maximum(self.topology_cells, new_connections)
            self.topology_cells = np.maximum(self.topology_cells, np.transpose(new_connections))
            topology_changed = int(self.topology_cells.sum()) > prev_edge_count

            # Update reward cells, refer to thesis for formulas.
            if topology_changed:
                reward_cells = np.where(self.reward_cells == 1, 1, 0)
                reward_cells_prior = reward_cells
                for t in range(1, 101):
                    reward_decay = 0.95 ** t  # Exponential decay over 100 hops
                    reward_cells = np.heaviside(np.dot(reward_cells_prior, self.topology_cells), 0)
                    reward_cells = np.maximum(reward_cells * reward_decay, reward_cells_prior)
                    reward_cells_prior = reward_cells
                self.reward_cells = reward_cells

        if entered_different_pc:
            self.prior_idx_pc_firing = idx_pc_active

    def get_enabled_neighbors(self, pc_idx):
        """Return indices of place cells connected to pc_idx in the topology."""
        if pc_idx >= self.topology_cells.shape[0]:
            return np.array([], dtype=np.int32)
        return np.where(self.topology_cells[pc_idx] == 1)[0]

    def path_exists(self, start_pc, goal_pc):
        """Check if a path exists from start_pc to goal_pc via BFS on topology."""
        if start_pc is None or goal_pc is None:
            return False
        n = self.topology_cells.shape[0]
        if start_pc >= n or goal_pc >= n or start_pc < 0 or goal_pc < 0:
            return False
        if start_pc == goal_pc:
            return True

        visited = set()
        queue = deque([start_pc])
        visited.add(start_pc)

        while queue:
            current = queue.popleft()
            neighbors = np.where(self.topology_cells[current] == 1)[0]
            for nb in neighbors:
                if nb == goal_pc:
                    return True
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return False

    def find_shortest_path(self, start_pc, goal_pc):
        """Find shortest path from start_pc to goal_pc via BFS on topology.

        Models hippocampal replay/preplay: the cognitive map graph encodes
        learned spatial relationships, and BFS finds the shortest sequence
        of place cells to traverse.
        """
        if start_pc is None or goal_pc is None:
            return []
        n = self.topology_cells.shape[0]
        if start_pc >= n or goal_pc >= n or start_pc < 0 or goal_pc < 0:
            return []
        if start_pc == goal_pc:
            return [start_pc]

        visited = set()
        queue = deque([start_pc])
        visited.add(start_pc)
        parent = {start_pc: None}

        while queue:
            current = queue.popleft()
            neighbors = np.where(self.topology_cells[current] == 1)[0]
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = current
                    if nb == goal_pc:
                        path = []
                        node = goal_pc
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        return path
                    queue.append(nb)
        return []

    def _update_reward_propagation(self):
        """Recalculate reward propagation through the topology after changes."""
        reward_cells = np.where(self.reward_cells == 1, 1, 0)
        reward_cells_prior = reward_cells
        for t in range(1, 101):
            reward_decay = 0.95 ** t  # Exponential decay over 100 hops
            reward_cells = np.heaviside(np.dot(reward_cells_prior, self.topology_cells), 0)
            reward_cells = np.maximum(reward_cells * reward_decay, reward_cells_prior)
            reward_cells_prior = reward_cells
        self.reward_cells = reward_cells

    def set_goal_pc_by_location(self, goal_location, pc_network):
        """Find the PC nearest to goal_location, set it as the sole reward=1 source,
        and re-propagate rewards through the topology."""
        best_idx = None
        best_dist = float('inf')
        for i, pc in enumerate(pc_network.place_cells):
            if pc.env_coordinates is not None:
                d = np.linalg.norm(pc.env_coordinates - np.array(goal_location))
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        if best_idx is not None:
            self.reward_cells = np.zeros_like(self.reward_cells)
            self.reward_cells[best_idx] = 1.0
            self._update_reward_propagation()
        return best_idx

    def save_cognitive_map(self, filename=""):
        directory = "data/cognitive_map/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save("data/cognitive_map/recency_cells" + filename + ".npy", self.recency_cells)
        np.save("data/cognitive_map/topology_cells" + filename + ".npy", self.topology_cells)
        np.save("data/cognitive_map/reward_cells" + filename + ".npy", self.reward_cells)

    def prune_blocked_connections(self, current_pc, agent_pos, directions, pc_network):
        """Remove topology connections where a wall blocks the path to a neighbor.

        If the direction toward a connected neighbor is blocked (wall detected
        by raycasts), the connection is pruned from the cognitive map so reward
        does not propagate through walls. Never prunes a connection that would
        disconnect the goal PC from the rest of the graph.
        """
        if current_pc is None or current_pc >= self.topology_cells.shape[0]:
            return False

        neighbors = np.where(self.topology_cells[current_pc] == 1)[0]
        if len(neighbors) == 0:
            return False

        goal_pc = np.argmax(self.reward_cells) if len(self.reward_cells) > 0 and np.max(self.reward_cells) > 0 else None

        num_dirs = len(directions)
        ray_angles = np.linspace(0, 2 * np.pi, num=num_dirs, endpoint=False)

        pruned = False
        for nb in neighbors:
            if nb >= len(pc_network.place_cells):
                continue
            nb_coords = pc_network.place_cells[nb].env_coordinates
            if nb_coords is None:
                continue

            # Angle from agent to neighbor PC
            delta = nb_coords - agent_pos
            angle_to_nb = np.arctan2(delta[1], delta[0])
            if angle_to_nb < 0:
                angle_to_nb += 2 * np.pi

            # Find nearest ray direction index (handles wrap-around)
            angle_diffs = np.abs(ray_angles - angle_to_nb)
            angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
            nearest_ray_idx = np.argmin(angle_diffs)

            # Also check adjacent rays (wall may be wide)
            left_idx = (nearest_ray_idx - 1) % num_dirs
            right_idx = (nearest_ray_idx + 1) % num_dirs

            # If the nearest direction AND both adjacent are blocked, the neighbor
            # is behind a wall — prune the connection.
            if (not directions[nearest_ray_idx]
                    and not directions[left_idx]
                    and not directions[right_idx]):
                dist_to_nb = np.linalg.norm(delta)
                # Only prune if neighbor is within ray range (~2-3m).
                if dist_to_nb < 3.0:
                    # Don't prune if it would disconnect the goal.
                    if goal_pc is not None:
                        self.topology_cells[current_pc][nb] = 0
                        self.topology_cells[nb][current_pc] = 0
                        still_connected = self.path_exists(current_pc, goal_pc) or self.path_exists(nb, goal_pc)
                        if not still_connected:
                            self.topology_cells[current_pc][nb] = 1
                            self.topology_cells[nb][current_pc] = 1
                            continue
                        if not self.path_exists(nb, goal_pc):
                            self.topology_cells[current_pc][nb] = 1
                            self.topology_cells[nb][current_pc] = 1
                            continue
                    else:
                        self.topology_cells[current_pc][nb] = 0
                        self.topology_cells[nb][current_pc] = 0

                    pass
                    pruned = True

        if pruned:
            self._update_reward_propagation()

        return pruned
