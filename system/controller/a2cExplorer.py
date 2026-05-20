import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class SpatialExplorationGrid:
    """Grid-based tracking of visited locations for spatial novelty."""

    def __init__(self, arena_size, cell_size=0.5, arena_bounds=None):
        if isinstance(arena_size, (int, float)):
            self.x_max = float(arena_size)
            self.y_max = float(arena_size)
        else:
            self.x_max = float(arena_size[0]) if len(arena_size) > 0 else 15.0
            self.y_max = float(arena_size[1]) if len(arena_size) > 1 else 15.0
        
        self.cell_size = cell_size
        self.nx = max(1, int(np.ceil(self.x_max / cell_size)))
        self.ny = max(1, int(np.ceil(self.y_max / cell_size)))

        self.visit_counts = np.zeros((self.nx, self.ny), dtype=np.int32)
        self.last_visit_step = np.full((self.nx, self.ny), -1, dtype=np.int32)

        # Pre-mark out-of-arena cells as heavily visited so they don't
        # create a false novelty gradient toward unreachable areas.
        if arena_bounds is not None:
            bx = arena_bounds[0] if isinstance(arena_bounds, (list, tuple)) else arena_bounds
            by = arena_bounds[1] if isinstance(arena_bounds, (list, tuple)) and len(arena_bounds) > 1 else bx
            bx_cell = int(np.ceil(bx / cell_size))
            by_cell = int(np.ceil(by / cell_size))
            blocked = 0
            for cx in range(self.nx):
                for cy in range(self.ny):
                    if cx >= bx_cell or cy >= by_cell:
                        self.visit_counts[cx, cy] = 9999
                        blocked += 1
            if blocked > 0:
                print(f"[SpatialGrid] Pre-blocked {blocked} out-of-bounds cells "
                      f"(arena bounds: {bx}x{by}m, grid: {self.nx}x{self.ny})")
        
        self.total_cells = self.nx * self.ny
    
    def _pos_to_cell(self, x, y):
        """Convert world position to grid cell indices."""
        cx = int(np.clip(x / self.cell_size, 0, self.nx - 1))
        cy = int(np.clip(y / self.cell_size, 0, self.ny - 1))
        return cx, cy

    def visit(self, x, y, step):
        """Record a visit to location (x, y) at step."""
        cx, cy = self._pos_to_cell(x, y)
        self.visit_counts[cx, cy] += 1
        self.last_visit_step[cx, cy] = step

    def get_novelty_info(self, x, y):
        """Return (visit_count, last_visit_step) for position."""
        cx, cy = self._pos_to_cell(x, y)
        return int(self.visit_counts[cx, cy]), int(self.last_visit_step[cx, cy])

    def compute_spatial_novelty(self, x, y, current_step, lambda_novelty=2.0,
                                lambda_recency=1.0, recency_decay=0.005):
        """Compute spatial novelty reward for a position."""
        visit_count, last_step = self.get_novelty_info(x, y)

        novelty_bonus = lambda_novelty / (1 + visit_count)

        if last_step >= 0:
            steps_since = current_step - last_step
            recency_bonus = lambda_recency * np.exp(-recency_decay * steps_since)
        else:
            recency_bonus = lambda_recency

        return novelty_bonus + recency_bonus

    def get_coverage_stats(self):
        """Get exploration coverage statistics."""
        visited_cells = np.sum(self.visit_counts > 0)
        coverage_pct = 100.0 * visited_cells / self.total_cells
        return {
            'visited_cells': int(visited_cells),
            'total_cells': self.total_cells,
            'coverage_pct': coverage_pct
        }

    def penalize_stuck_location(self, x, y, radius=2.0, base_penalty=200):
        """Mark cells around a stuck position as heavily visited."""
        if not hasattr(self, 'stuck_history'):
            self.stuck_history = []
            self.wall_zones = []

        nearby_count = 0
        for sx, sy in self.stuck_history:
            if np.sqrt((x - sx)**2 + (y - sy)**2) < 2.0:
                nearby_count += 1
        self.stuck_history.append((x, y))

        # Escalating penalty: doubles with each repeat (capped at 16x)
        escalation = min(2 ** nearby_count, 16)
        penalty = base_penalty * escalation

        # After 3+ stuck events in same area, mark as wall zone
        if nearby_count >= 2:
            already_marked = False
            for wx, wy, wr in self.wall_zones:
                if np.sqrt((x - wx)**2 + (y - wy)**2) < 1.5:
                    already_marked = True
                    break
            if not already_marked:
                self.wall_zones.append((x, y, radius))
                if nearby_count >= 2:
                    print(f"[WALL ZONE] Marked implied wall at ({x:.1f},{y:.1f}) after {nearby_count+1} stuck events")

        cx, cy = self._pos_to_cell(x, y)
        r_cells = int(np.ceil(radius / self.cell_size))

        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                nx_cell = cx + dx
                ny_cell = cy + dy
                if 0 <= nx_cell < self.nx and 0 <= ny_cell < self.ny:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= r_cells:
                        scale = 1.0 - (dist / (r_cells + 1))
                        self.visit_counts[nx_cell, ny_cell] += int(penalty * scale)

    def compute_novelty_direction(self, x, y, radius=3.0):
        """Compute a direction vector pointing toward less-visited nearby cells."""
        cx, cy = self._pos_to_cell(x, y)
        r_cells = int(np.ceil(radius / self.cell_size))

        wall_zones = getattr(self, 'wall_zones', [])

        direction = np.array([0.0, 0.0])

        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                nx_cell = cx + dx
                ny_cell = cy + dy

                if nx_cell < 0 or nx_cell >= self.nx or ny_cell < 0 or ny_cell >= self.ny:
                    continue

                if dx == 0 and dy == 0:
                    continue

                dist_cells = np.sqrt(dx**2 + dy**2)
                if dist_cells > r_cells:
                    continue

                cell_world_x = (nx_cell + 0.5) * self.cell_size
                cell_world_y = (ny_cell + 0.5) * self.cell_size

                # Skip cells on the far side of a wall zone from the agent.
                blocked = False
                for wx, wy, wr in wall_zones:
                    dx_aw = wx - x
                    dy_aw = wy - y
                    dx_ac = cell_world_x - x
                    dy_ac = cell_world_y - y

                    line_len_sq = dx_ac**2 + dy_ac**2
                    if line_len_sq > 0.01:
                        t = (dx_aw * dx_ac + dy_aw * dy_ac) / line_len_sq
                        if 0.1 < t < 0.9:
                            proj_x = x + t * dx_ac
                            proj_y = y + t * dy_ac
                            perp_dist = np.sqrt((wx - proj_x)**2 + (wy - proj_y)**2)
                            if perp_dist < wr * 0.8:
                                blocked = True
                                break

                if blocked:
                    continue

                visit_count = self.visit_counts[nx_cell, ny_cell]
                novelty = 1.0 / (1.0 + visit_count)

                # Flat distance falloff so distant unvisited areas still attract strongly.
                dist_weight = 1.0 / (1.0 + 0.1 * dist_cells)

                dir_vec = np.array([cell_world_x - x, cell_world_y - y])
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > 0.01:
                    dir_vec = dir_vec / dir_norm

                direction += dir_vec * novelty * dist_weight

        norm = np.linalg.norm(direction)
        if norm > 0.01:
            direction = direction / norm

        return direction


def compute_intrinsic_reward(visit_count, last_visit_step, current_step,
                             lambda_novelty=1.0, lambda_recency=0.5, recency_decay=0.01):
    """Pure function to compute intrinsic curiosity reward (PC-based).

    Novelty: 1 / (1 + visit_count). Recency: exp(-alpha * steps_since_visit).
    """
    novelty_bonus = lambda_novelty / (1 + visit_count)

    if last_visit_step >= 0:
        steps_since_visit = current_step - last_visit_step
        recency_bonus = lambda_recency * np.exp(-recency_decay * steps_since_visit)
    else:
        recency_bonus = lambda_recency

    return novelty_bonus + recency_bonus


def build_exploration_state(robot_pose, cognitive_map, current_pc, arena_size=15.0,
                            goal_location=None):

    base_dim = 13
    if current_pc is None or current_pc < 0:
        return torch.zeros(base_dim, dtype=torch.float32)

    nr_pcs = cognitive_map.nr_place_cells

    neighbors = cognitive_map.get_enabled_neighbors(current_pc)

    valid_neighbors = [n for n in neighbors if n < len(cognitive_map.visit_counts)]
    neighbors = np.array(valid_neighbors, dtype=np.int32)

    visit_count = cognitive_map.visit_counts[current_pc] if current_pc < len(cognitive_map.visit_counts) else 0
    recency = cognitive_map.last_visit_step[current_pc] if current_pc < len(cognitive_map.last_visit_step) else -1

    if len(neighbors) > 0:
        neighbor_visits = cognitive_map.visit_counts[neighbors]
        neighbor_recency = cognitive_map.last_visit_step[neighbors]
    else:
        neighbor_visits = np.array([0])
        neighbor_recency = np.array([-1])

    # Pose normalization
    x_norm = robot_pose[0] / arena_size
    y_norm = robot_pose[1] / arena_size
    theta_norm = robot_pose[2] / np.pi

    pc_norm = current_pc / max(nr_pcs, 1)

    # Visit counts: log-scale for stability
    visit_norm = np.log1p(float(visit_count))
    neighbor_visits_norm = np.log1p(neighbor_visits.astype(np.float32))

    # Recency: -1 (never visited) maps to 0; soft cap at 1000 steps.
    recency_norm = max(float(recency), 0) / 1000.0
    neighbor_recency_norm = np.maximum(neighbor_recency.astype(np.float32), 0) / 1000.0

    neighbor_count_norm = len(neighbors) / max(nr_pcs, 1)
    mean_neighbor_visits = float(neighbor_visits_norm.mean()) if len(neighbor_visits_norm) > 0 else 0.0
    max_neighbor_recency = float(neighbor_recency_norm.max()) if len(neighbor_recency_norm) > 0 else 0.0

    if np.isnan(mean_neighbor_visits):
        mean_neighbor_visits = 0.0
    if np.isnan(max_neighbor_recency):
        max_neighbor_recency = 0.0

    # Agent-frame goal vector (rotated by heading) + distance + reachability.
    if goal_location is not None:
        gx, gy = float(goal_location[0]), float(goal_location[1])
        dx = gx - float(robot_pose[0])
        dy = gy - float(robot_pose[1])
        # Rotate world (dx, dy) into agent frame using -theta
        theta = float(robot_pose[2])
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        goal_forward = (cos_t * dx + sin_t * dy) / arena_size
        goal_left = (-sin_t * dx + cos_t * dy) / arena_size
        goal_dist_norm = float(np.clip(np.sqrt(dx * dx + dy * dy) / arena_size, 0.0, 1.0))
        goal_pc = int(np.argmax(cognitive_map.reward_cells)) if len(cognitive_map.reward_cells) > 0 else -1
        has_path_flag = 1.0 if (goal_pc >= 0 and cognitive_map.path_exists(current_pc, goal_pc)) else 0.0
    else:
        goal_forward = 0.0
        goal_left = 0.0
        goal_dist_norm = 0.0
        has_path_flag = 0.0

    state = np.array([
        x_norm, y_norm, theta_norm,
        pc_norm, visit_norm, recency_norm,
        neighbor_count_norm,
        mean_neighbor_visits,
        max_neighbor_recency,
        goal_forward, goal_left,
        goal_dist_norm, has_path_flag
    ], dtype=np.float32)

    if np.any(np.isnan(state)):
        state = np.nan_to_num(state, nan=0.0)

    return torch.tensor(state, dtype=torch.float32)


def action_to_goal_vector(action, heading=0.0, step_size=1.0):
    """Convert discrete action to goal vector in global coordinates.

    Actions are defined in robot-relative frame and rotated by heading
    into global frame.
        0 = forward, 1 = left, 2 = right, 3 = backward, 4 = stop
    """
    local_vectors = {
        0: np.array([step_size, 0]),
        1: np.array([0, step_size]),
        2: np.array([0, -step_size]),
        3: np.array([-step_size, 0]),
        4: np.array([0, 0]),
    }
    local_vec = local_vectors.get(action, np.array([0, 0]))

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    global_vec = np.array([
        cos_h * local_vec[0] - sin_h * local_vec[1],
        sin_h * local_vec[0] + cos_h * local_vec[1]
    ])
    return global_vec


class ActorCritic(nn.Module):
    """Combined Actor-Critic network for A2C."""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value


class A2CExplorer:
    """A2C-based exploration policy for curiosity-driven navigation."""

    def __init__(self, state_dim=13, action_dim=5, lr=0.0003, gamma=0.99,
                 entropy_coef=0.01, value_coef=0.5):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.training_steps = 0

    def act(self, state: torch.Tensor):
        """Select action from policy. Returns (action, log_prob, value)."""
        with torch.no_grad():
            action_probs, value = self.network(state)

        # NaN protection: if action_probs has NaN, fall back to uniform.
        if torch.any(torch.isnan(action_probs)):
            print(f"[A2C WARNING] NaN in action_probs, using uniform. State: {state}")
            action_probs = torch.ones(action_probs.shape) / action_probs.shape[-1]

        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value.squeeze()

    def _compute_returns(self, rewards, gamma):
        """Compute discounted returns from rewards."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, transitions):
        """A2C update from collected transitions."""
        if len(transitions) == 0:
            return {'loss': 0, 'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

        states = torch.stack([t['state'] for t in transitions])
        actions = torch.tensor([t['action'] for t in transitions])
        old_log_probs = torch.stack([t['log_prob'] for t in transitions])
        values = torch.stack([t['value'] for t in transitions])
        rewards = [t['reward'] for t in transitions]

        returns = self._compute_returns(rewards, self.gamma)
        advantages = returns - values.detach()

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_probs, current_values = self.network(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * advantages).mean()

        value_loss = nn.functional.mse_loss(current_values.squeeze(), returns)

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.training_steps += 1

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint.get('training_steps', 0)
