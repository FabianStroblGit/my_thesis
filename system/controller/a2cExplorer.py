import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ============================================================================
# Spatial Exploration Grid (for true spatial novelty tracking)
# ============================================================================

class SpatialExplorationGrid:
    """Grid-based tracking of visited locations for spatial novelty.
    
    Unlike PC-based visit counting (which rewards creating new PCs in the same area),
    this tracks actual spatial coverage by dividing the arena into fixed-size cells.
    """
    
    def __init__(self, arena_size, cell_size=0.5, arena_bounds=None):
        """Initialize spatial grid.
        
        Args:
            arena_size: tuple (x_max, y_max) or float for square arena (grid extent)
            cell_size: float, size of each grid cell in meters
            arena_bounds: tuple (x_max, y_max) of actual reachable area.
                         If provided, cells outside these bounds are pre-marked
                         as visited so they don't create a false novelty gradient.
        """
        if isinstance(arena_size, (int, float)):
            self.x_max = float(arena_size)
            self.y_max = float(arena_size)
        else:
            self.x_max = float(arena_size[0]) if len(arena_size) > 0 else 15.0
            self.y_max = float(arena_size[1]) if len(arena_size) > 1 else 15.0
        
        self.cell_size = cell_size
        self.nx = max(1, int(np.ceil(self.x_max / cell_size)))
        self.ny = max(1, int(np.ceil(self.y_max / cell_size)))
        
        # Visit count grid (shared among all agents)
        self.visit_counts = np.zeros((self.nx, self.ny), dtype=np.int32)
        # Last visit step for recency
        self.last_visit_step = np.full((self.nx, self.ny), -1, dtype=np.int32)
        
        # Pre-mark cells outside actual arena bounds as heavily visited
        # so they don't create a false novelty gradient pulling agents
        # toward unreachable areas outside the walls
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
        """Get visit count and last visit step for position.
        
        Returns:
            tuple: (visit_count, last_visit_step)
        """
        cx, cy = self._pos_to_cell(x, y)
        return int(self.visit_counts[cx, cy]), int(self.last_visit_step[cx, cy])
    
    def compute_spatial_novelty(self, x, y, current_step, lambda_novelty=2.0, 
                                lambda_recency=1.0, recency_decay=0.005):
        """Compute spatial novelty reward for a position.
        
        Higher weights than PC-based novelty to encourage actual movement.
        """
        visit_count, last_step = self.get_novelty_info(x, y)
        
        # Novelty: unvisited cells get high reward
        novelty_bonus = lambda_novelty / (1 + visit_count)
        
        # Recency: recently visited cells get lower reward
        if last_step >= 0:
            steps_since = current_step - last_step
            recency_bonus = lambda_recency * np.exp(-recency_decay * steps_since)
        else:
            recency_bonus = lambda_recency  # Never visited
        
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
        """Mark cells around a stuck position as heavily visited.
        
        
        Args:
            x, y: position where agent got stuck
            radius: radius in meters to penalize around stuck point
            base_penalty: base visit count to add (multiplied by repeat count)
        """
        # Track stuck history for escalation
        if not hasattr(self, 'stuck_history'):
            self.stuck_history = []  # list of (x, y) stuck positions
            self.wall_zones = []     # list of (x, y, radius) marking implied walls
        
        # Count how many times agent got stuck near this location
        nearby_count = 0
        for sx, sy in self.stuck_history:
            if np.sqrt((x - sx)**2 + (y - sy)**2) < 2.0:
                nearby_count += 1
        self.stuck_history.append((x, y))
        
        # Escalating penalty: doubles with each repeat
        escalation = min(2 ** nearby_count, 16)  # Cap at 16x
        penalty = base_penalty * escalation
        
        # After 3+ stuck events in same area, mark as wall zone
        if nearby_count >= 2:
            # Add wall zone to block novelty gradient from pulling through here
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
                        # Stronger penalty at center, weaker at edges
                        scale = 1.0 - (dist / (r_cells + 1))
                        self.visit_counts[nx_cell, ny_cell] += int(penalty * scale)
    
    def compute_novelty_direction(self, x, y, radius=3.0):
        """Compute a direction vector pointing toward less-visited nearby cells.
        
        Args:
            x, y: current position
            radius: search radius in meters (how far to look for novel cells)
            
        Returns:
            np.ndarray: normalized 2D direction vector toward least-visited area
        """
        cx, cy = self._pos_to_cell(x, y)
        r_cells = int(np.ceil(radius / self.cell_size))
        
        # Collect wall zones for filtering
        wall_zones = getattr(self, 'wall_zones', [])
        
        # Weighted direction sum: each cell contributes a vector proportional to its novelty
        direction = np.array([0.0, 0.0])
        
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                nx_cell = cx + dx
                ny_cell = cy + dy
                
                # Skip out-of-bounds cells
                if nx_cell < 0 or nx_cell >= self.nx or ny_cell < 0 or ny_cell >= self.ny:
                    continue
                
                # Skip the agent's own cell
                if dx == 0 and dy == 0:
                    continue
                
                # Distance in cells
                dist_cells = np.sqrt(dx**2 + dy**2)
                if dist_cells > r_cells:
                    continue
                
                # World position of this cell center
                cell_world_x = (nx_cell + 0.5) * self.cell_size
                cell_world_y = (ny_cell + 0.5) * self.cell_size
                
                # Wall-aware filtering: skip cells that are beyond a wall zone
                # A cell is "beyond a wall" if the line from agent to cell
                # passes through or near a wall zone
                blocked = False
                for wx, wy, wr in wall_zones:
                    # Check if the target cell is on the far side of the wall from agent
                    # Simple check: if wall is between agent and cell (perpendicular distance)
                    dx_aw = wx - x
                    dy_aw = wy - y
                    dx_ac = cell_world_x - x
                    dy_ac = cell_world_y - y
                    
                    # Project wall position onto agent-to-cell line
                    line_len_sq = dx_ac**2 + dy_ac**2
                    if line_len_sq > 0.01:
                        t = (dx_aw * dx_ac + dy_aw * dy_ac) / line_len_sq
                        if 0.1 < t < 0.9:  # Wall is between agent and cell
                            # Perpendicular distance from wall to line
                            proj_x = x + t * dx_ac
                            proj_y = y + t * dy_ac
                            perp_dist = np.sqrt((wx - proj_x)**2 + (wy - proj_y)**2)
                            if perp_dist < wr * 0.8:  # Close to wall zone
                                blocked = True
                                break
                
                if blocked:
                    continue
                
                # Novelty weight: unvisited cells are most attractive
                visit_count = self.visit_counts[nx_cell, ny_cell]
                novelty = 1.0 / (1.0 + visit_count)
                
                # Flat distance falloff so distant unvisited areas still attract strongly
                # (was 1/(1+dist_cells) which made cells 10 away have only 9% weight)
                dist_weight = 1.0 / (1.0 + 0.1 * dist_cells)
                
                # Direction vector from agent to this cell
                dir_vec = np.array([cell_world_x - x, cell_world_y - y])
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > 0.01:
                    dir_vec = dir_vec / dir_norm  # normalize direction
                
                # Accumulate weighted direction
                direction += dir_vec * novelty * dist_weight
        
        # Normalize final direction
        norm = np.linalg.norm(direction)
        if norm > 0.01:
            direction = direction / norm
        
        return direction


# ============================================================================
# Intrinsic Reward (pure function, separate from A2C class)
# ============================================================================

def compute_intrinsic_reward(visit_count, last_visit_step, current_step,
                             lambda_novelty=1.0, lambda_recency=0.5, recency_decay=0.01):
    """Pure function to compute intrinsic curiosity reward (PC-based).
    
    Uses bounded forms for numerical stability:
    - Novelty: 1 / (1 + visit_count) — decays as PC is visited more
    - Recency: exp(-α * steps_since_visit) — bounded [0, 1], stable over long runs
    
    Args:
        visit_count: int, number of times the current PC has been visited
        last_visit_step: int, step number when PC was last visited (-1 if never)
        current_step: int, current simulation step
        lambda_novelty: float, weight for novelty bonus
        lambda_recency: float, weight for recency bonus
        recency_decay: float, decay rate for recency (higher = faster decay)
    
    Returns:
        float: intrinsic reward value
    """
    novelty_bonus = lambda_novelty / (1 + visit_count)
    
    if last_visit_step >= 0:
        steps_since_visit = current_step - last_visit_step
        recency_bonus = lambda_recency * np.exp(-recency_decay * steps_since_visit)
    else:
        recency_bonus = lambda_recency  # Never visited → full recency bonus
    
    return novelty_bonus + recency_bonus


# ============================================================================
# State Builder
# ============================================================================

def build_exploration_state(robot_pose, cognitive_map, current_pc, arena_size=15.0,
                            goal_location=None):

    base_dim = 13
    if current_pc is None or current_pc < 0:
        # Return zero state if no valid PC
        return torch.zeros(base_dim, dtype=torch.float32)
    
    nr_pcs = cognitive_map.nr_place_cells
    
    # Get enabled neighbors (consistent with path_exists, respects blocked edges)
    neighbors = cognitive_map.get_enabled_neighbors(current_pc)
    
    # Filter neighbors to only include valid indices for visit tracking arrays
    valid_neighbors = [n for n in neighbors if n < len(cognitive_map.visit_counts)]
    neighbors = np.array(valid_neighbors, dtype=np.int32)
    
    # Current PC features
    visit_count = cognitive_map.visit_counts[current_pc] if current_pc < len(cognitive_map.visit_counts) else 0
    recency = cognitive_map.last_visit_step[current_pc] if current_pc < len(cognitive_map.last_visit_step) else -1
    
    # Neighbor features (aggregated)
    if len(neighbors) > 0:
        neighbor_visits = cognitive_map.visit_counts[neighbors]
        neighbor_recency = cognitive_map.last_visit_step[neighbors]
    else:
        neighbor_visits = np.array([0])
        neighbor_recency = np.array([-1])
    
    # === Normalization ===
    # Pose: divide by arena size, theta already in [-pi, pi]
    x_norm = robot_pose[0] / arena_size
    y_norm = robot_pose[1] / arena_size
    theta_norm = robot_pose[2] / np.pi  # [-1, 1]
    
    # PC index: normalize to [0, 1]
    pc_norm = current_pc / max(nr_pcs, 1)  # Avoid div by zero
    
    # Visit counts: log-scale for stability (log(1 + count))
    visit_norm = np.log1p(float(visit_count))
    neighbor_visits_norm = np.log1p(neighbor_visits.astype(np.float32))
    
    # Recency: normalize by max possible steps or use relative
    # -1 means never visited -> map to 0
    recency_norm = max(float(recency), 0) / 1000.0  # Soft cap at 1000 steps
    neighbor_recency_norm = np.maximum(neighbor_recency.astype(np.float32), 0) / 1000.0
    
    # Compute aggregated neighbor features with NaN protection
    neighbor_count_norm = len(neighbors) / max(nr_pcs, 1)
    mean_neighbor_visits = float(neighbor_visits_norm.mean()) if len(neighbor_visits_norm) > 0 else 0.0
    max_neighbor_recency = float(neighbor_recency_norm.max()) if len(neighbor_recency_norm) > 0 else 0.0
    
    # Ensure no NaN values
    if np.isnan(mean_neighbor_visits):
        mean_neighbor_visits = 0.0
    if np.isnan(max_neighbor_recency):
        max_neighbor_recency = 0.0
    
    # === Goal features ===
    # Agent-frame goal vector (rotated by heading): goal_forward/left tell the
    # policy "how far ahead and to the side is the goal". Distance and reachability
    # tell the policy "are we close" and "is the cognitive map connecting us".
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
        # Reachability via topology
        goal_pc = int(np.argmax(cognitive_map.reward_cells)) if len(cognitive_map.reward_cells) > 0 else -1
        has_path_flag = 1.0 if (goal_pc >= 0 and cognitive_map.path_exists(current_pc, goal_pc)) else 0.0
    else:
        goal_forward = 0.0
        goal_left = 0.0
        goal_dist_norm = 0.0
        has_path_flag = 0.0

    state = np.array([
        x_norm, y_norm, theta_norm,                    # (3,) normalized pose
        pc_norm, visit_norm, recency_norm,             # (3,) current PC info
        neighbor_count_norm,                            # neighbor count normalized
        mean_neighbor_visits,                           # mean log-visit count
        max_neighbor_recency,                           # max recency of neighbors
        goal_forward, goal_left,                        # (2,) agent-frame goal vector
        goal_dist_norm, has_path_flag                   # (2,) distance + reachability
    ], dtype=np.float32)

    # Final NaN check
    if np.any(np.isnan(state)):
        state = np.nan_to_num(state, nan=0.0)

    return torch.tensor(state, dtype=torch.float32)


def action_to_goal_vector(action, heading=0.0, step_size=1.0):
    """Convert discrete action to goal vector in global coordinates.
    
    Actions are in robot-relative frame and rotated by heading to global frame.
    
    Args:
        action: int, discrete action index
            0 = forward (robot's forward direction)
            1 = left (robot's left)
            2 = right (robot's right)
            3 = backward (robot's backward)
            4 = stop
        heading: float, robot heading in radians
        step_size: float, magnitude of movement (increased to 1.0 for more movement)
    
    Returns:
        np.ndarray: 2D goal vector in global coordinates
    """
    # Actions are defined in robot-relative frame
    local_vectors = {
        0: np.array([step_size, 0]),      # forward (robot's forward direction)
        1: np.array([0, step_size]),      # left (robot's left)
        2: np.array([0, -step_size]),     # right (robot's right)
        3: np.array([-step_size, 0]),     # backward (robot's backward)
        4: np.array([0, 0]),              # stop
    }
    local_vec = local_vectors.get(action, np.array([0, 0]))
    
    # Rotate from robot frame to global frame
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    global_vec = np.array([
        cos_h * local_vec[0] - sin_h * local_vec[1],
        sin_h * local_vec[0] + cos_h * local_vec[1]
    ])
    return global_vec


# ============================================================================
# A2C Explorer Neural Network
# ============================================================================

class ActorCritic(nn.Module):
    """Combined Actor-Critic network for A2C."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
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
        """Initialize A2C explorer.
        
        Args:
            state_dim: int, dimension of state vector (default 9)
            action_dim: int, number of discrete actions (default 5)
            lr: float, learning rate
            gamma: float, discount factor
            entropy_coef: float, entropy bonus coefficient
            value_coef: float, value loss coefficient
        """
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # For tracking
        self.training_steps = 0
    
    def act(self, state: torch.Tensor):
        """Select action from policy.
        
        Args:
            state: torch.Tensor of shape (state_dim,) — already a tensor from build_exploration_state
        
        Returns:
            action: int, discrete action index
            log_prob: torch.Tensor, log probability of action
            value: torch.Tensor, critic value estimate
        """
        # No tensor conversion needed — state is already torch.Tensor
        with torch.no_grad():
            action_probs, value = self.network(state)
        
        # NaN protection: if action_probs has NaN, use uniform distribution
        if torch.any(torch.isnan(action_probs)):
            print(f"[A2C WARNING] NaN in action_probs, using uniform. State: {state}")
            action_probs = torch.ones(action_probs.shape) / action_probs.shape[-1]
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()
    
    def _compute_returns(self, rewards, gamma):
        """Compute discounted returns from rewards.
        
        Args:
            rewards: list of float, rewards for each step
            gamma: float, discount factor
        
        Returns:
            torch.Tensor: discounted returns
        """
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)
    
    def update(self, transitions):
        """A2C update from collected transitions.
        
        Args:
            transitions: list of dicts with keys: state, action, log_prob, value, reward
        
        Returns:
            dict: training statistics
        """
        if len(transitions) == 0:
            return {'loss': 0, 'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
        
        states = torch.stack([t['state'] for t in transitions])
        actions = torch.tensor([t['action'] for t in transitions])
        old_log_probs = torch.stack([t['log_prob'] for t in transitions])
        values = torch.stack([t['value'] for t in transitions])
        rewards = [t['reward'] for t in transitions]
        
        # Compute discounted returns
        returns = self._compute_returns(rewards, self.gamma)
        advantages = returns - values.detach()
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass to get current policy
        action_probs, current_values = self.network(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Policy loss (actor)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss (critic)
        value_loss = nn.functional.mse_loss(current_values.squeeze(), returns)
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Backpropagation
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
