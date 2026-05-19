import pybullet as p
import time
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pybullet_data
import numpy as np

from system.helper import compute_angle

from system.controller.navigationPhase import pick_intermediate_goal_vector, find_new_goal_vector


class PybulletEnvironment:
    """This class deals with everything pybullet or environment (obstacles) related"""
    def __init__(self, visualize, env_model, dt, pod=None, doors_option="plane_doors", camera_config=None):
        self.visualize = visualize  # to open JAVA application
        self.env_model = env_model  # string specifying env_model
        self.pod = pod  # if Phase Offset detectors are used
        self.doors_option = doors_option  # valid: plane, plane_doors, plane_doors_individual
        self.door_positions = []
        self.camera_config = camera_config or {}
        self.camera_enabled = bool(self.camera_config.get("enabled", False))
        self.camera_width = int(self.camera_config.get("width", 320))
        self.camera_height = int(self.camera_config.get("height", 240))
        self.camera_fov = float(self.camera_config.get("fov", 60.0))
        self.camera_near = float(self.camera_config.get("near", 0.1))
        self.camera_far = float(self.camera_config.get("far", 10.0))
        self.camera_attach_to_agent = bool(self.camera_config.get("attach_to_agent", False))
        self.camera_relative_position = np.array(self.camera_config.get("relative_position", [0.2, 0.0, 0.3]),
                            dtype=float)
        self.camera_relative_target = np.array(self.camera_config.get("relative_target", [1.0, 0.0, 0.0]),
                              dtype=float)
        self.camera_relative_up = np.array(self.camera_config.get("relative_up", [0.0, 0.0, 1.0]), dtype=float)
        self.camera_position = np.array(self.camera_config.get("position", [5.5, 0.5, 3.5]), dtype=float)
        self.camera_target = np.array(self.camera_config.get("target", [5.5, 5.5, 0.0]), dtype=float)
        self.camera_up_vector = np.array(self.camera_config.get("up", [0.0, 0.0, 1.0]), dtype=float)
        self.camera_view_matrix = None
        self.camera_projection_matrix = None
        self.latest_camera_frame = None
        self.camera_frame_index = 0

        if self.visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        base_position = [0, 0.05, 0.02]  # [0, 0.05, 0.02] ensures that it actually starts at origin
        arena_size = 15  # circular arena size with radius r
        goal_location = None
        max_speed = 5.5  # determines speed at which agent travels: max_speed = 5.5 -> actual speed of ~0.5 m/s

        if env_model == "plus":
            p.loadURDF("p3dx/plane/plane.urdf")
        elif env_model == "obstacle":
            p.loadURDF("environment/obstacle_map/plane.urdf")
        elif env_model == "linear_sunburst":
            p.loadURDF("environment/linear_sunburst_map/" + self.doors_option + ".urdf")
            self.door_positions = self._load_linear_sunburst_doors(self.doors_option)
            base_position = [5.5, 0.55, 0.02]
            arena_size = 15
            goal_location = np.array([1.5, 10])
            max_speed = 6
        else:
            urdfRootPath = pybullet_data.getDataPath()
            p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"))

        orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])  # faces North

        self.carID = p.loadURDF("p3dx/urdf/pioneer3dx.urdf", basePosition=base_position, baseOrientation=orientation)

        p.setGravity(0, 0, -9.81)

        self.dt = dt
        p.setTimeStep(self.dt)

        self.xy_coordinates = []  # keeps track of agent's coordinates at each time step
        self.orientation_angle = []  # keeps track of agent's orientation at each time step
        self.xy_speeds = []  # keeps track of agent's speed (vector) at each time step
        self.speeds = []  # keeps track of agent's speed (value) at each time step
        self.save_position_and_speed()  # save initial configuration

        if self.camera_enabled:
            self._initialize_camera()
            self.capture_camera_frame()

        # Set goal location to preset location or current position if none was specified
        self.goal_location = goal_location if goal_location is not None else self.xy_coordinates[0]

        self.max_speed = max_speed
        self.arena_size = arena_size
        self.goal = np.array([0, 0])  # used for navigation (eg. sub goals)

        self.goal_vector_original = np.array([1, 1])  # egocentric goal vector after last recalculation
        self.goal_vector = np.array([0, 0])  # egocentric goal vector after last update

        self.goal_idx = 0  # pc_idx of goal

        self.turning = False  # agent state, used for controller

        self.num_ray_dir = 16  # number of direction to check for obstacles for
        self.num_travel_dir = 2  # valid traveling directions, 2 -> every 2nd of 16 = 8 dirs (E,NE,N,NW,W,SW,S,SE)
        self.directions = np.empty(self.num_ray_dir, dtype=bool)  # array keeping track which directions are blocked
        self.topology_based = False  # agent state, used for controller

        # Track current PC for cognitive map updates
        self.current_pc_idx = None  # PC index the agent is currently at
        self.target_pc_idx = None  # PC index the agent is trying to reach

        # A2C exploration state (used when no path exists to goal)
        self.exploration_mode = False       # True when using A2C exploration
        self.exploration_step_count = 0     # Steps spent in current exploration bout
        self.a2c_action_repeat = 200         # How many steps to persist one A2C action
        self.a2c_action_counter = 0         # Steps remaining for current A2C action
        self.a2c_current_action = None      # Current A2C action index (0-4)
        self.a2c_goal_vector = None         # Goal vector from current A2C action
        self.path_recheck_interval = 50     # Re-check path_exists every N steps
        self.pending_transition = None      # Pending A2C transition for training

        # Thigmotaxis (wall-following) state
        self.wall_follow_active = False     # True when in wall-following mode
        self.wall_follow_direction = 1      # +1 = follow wall to the right, -1 = to the left
        self.wall_follow_steps = 0          # Steps spent wall-following in current direction
        self.wall_follow_max_steps = 3000   # Max steps before reversing direction
        self.wall_follow_reversals = 0      # Number of direction reversals

        # Debug text IDs for reward visualization
        self.reward_text_ids = []

    def update_reward_visualization(self, pc_network, cognitive_map):
        """Display reward values at each place cell location in the PyBullet visualization."""
        if not self.visualize:
            return
        
        # Remove old text labels
        for text_id in self.reward_text_ids:
            p.removeUserDebugItem(text_id)
        self.reward_text_ids = []
        
        # Add new text labels for each place cell
        for i, pc in enumerate(pc_network.place_cells):
            if pc.env_coordinates is None:
                continue
            
            reward_value = cognitive_map.reward_cells[i]
            if reward_value > 0:  # Only show non-zero rewards
                x, y = pc.env_coordinates
                # Color from red (low reward) to green (high reward)
                color = [1 - reward_value, reward_value, 0]  # RGB
                text_id = p.addUserDebugText(
                    text=f"{reward_value:.2f}",
                    textPosition=[x, y, 0.3],  # Slightly above ground
                    textColorRGB=color,
                    textSize=1.0
                )
                self.reward_text_ids.append(text_id)

    @staticmethod
    def _load_linear_sunburst_doors(doors_option):
        """Extract door positions from the selected URDF to keep plots in sync with environment."""
        urdf_path = Path("environment/linear_sunburst_map") / f"{doors_option}.urdf"
        if not urdf_path.exists():
            return []

        door_positions = []
        try:
            tree = ET.parse(urdf_path)
        except ET.ParseError:
            return []

        for link in tree.findall("link"):
            name = link.attrib.get("name", "")
            if not name.startswith("door"):
                continue
            origin = link.find("visual/origin")
            if origin is None:
                continue
            xyz = origin.attrib.get("xyz", "")
            try:
                x_coord = float(xyz.split()[0])
            except (ValueError, IndexError):
                continue
            door_positions.append(round(x_coord - 0.5, 3))

        return sorted(door_positions)

    def _initialize_camera(self):
        """Configure projection matrix and prime the initial camera view."""
        aspect = (self.camera_width / self.camera_height) if self.camera_height else 1.0
        self.camera_projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=aspect,
            nearVal=self.camera_near,
            farVal=self.camera_far
        )
        self._update_camera_view_matrix()

    def _compute_camera_transforms(self):
        if self.camera_attach_to_agent:
            base_position, base_orientation = p.getBasePositionAndOrientation(self.carID)
            base_position = np.array(base_position)
            rot_matrix = np.array(p.getMatrixFromQuaternion(base_orientation)).reshape(3, 3)

            eye = base_position + rot_matrix.dot(self.camera_relative_position)
            target = base_position + rot_matrix.dot(self.camera_relative_target)
            up_vector = rot_matrix.dot(self.camera_relative_up)

            norm = np.linalg.norm(up_vector)
            if norm > 1e-6:
                up_vector = up_vector / norm

            return eye.tolist(), target.tolist(), up_vector.tolist()

        return (self.camera_position.tolist(),
                self.camera_target.tolist(),
                self.camera_up_vector.tolist())

    def _update_camera_view_matrix(self):
        if not self.camera_enabled:
            return

        eye, target, up_vector = self._compute_camera_transforms()
        self.camera_view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=up_vector
        )

    def capture_camera_frame(self):
        """Capture camera RGB, depth, and segmentation arrays if enabled."""
        if not self.camera_enabled:
            return None

        if self.camera_projection_matrix is None:
            self._initialize_camera()

        self._update_camera_view_matrix()
        if self.camera_view_matrix is None:
            return None

        width, height, rgb_pixels, depth_pixels, segmentation_pixels = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=self.camera_view_matrix,
            projectionMatrix=self.camera_projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )

        rgb_array = np.reshape(np.array(rgb_pixels, dtype=np.uint8), (height, width, 4))
        depth_array = np.reshape(np.array(depth_pixels, dtype=np.float32), (height, width))
        segmentation_array = np.reshape(np.array(segmentation_pixels, dtype=np.int32), (height, width))

        frame = {
            "index": self.camera_frame_index,
            "rgb": rgb_array,
            "depth": depth_array,
            "segmentation": segmentation_array,
            "timestamp": (len(self.xy_coordinates) - 1) * self.dt
        }

        self.latest_camera_frame = frame
        self.camera_frame_index += 1
        return frame

    def get_latest_camera_frame(self):
        """Return most recent camera frame for downstream processing."""
        return self.latest_camera_frame

    def detect_door_opening(self, wall_threshold=2.0, min_opening_width=15):
        """Detect door openings using depth camera image.
        
        Analyzes the depth image to find gaps/openings in walls by looking for
        regions where depth suddenly increases (indicating a passage through).
        
        Args:
            wall_threshold: float, depth below this is considered a wall (meters)
            min_opening_width: int, minimum pixel width of opening to consider
            
        Returns:
            tuple: (door_angle, door_distance, confidence) or (None, None, 0) if no door found
                - door_angle: angle in radians relative to robot heading (positive = left)
                - door_distance: estimated distance to door opening
                - confidence: 0-1 score of detection confidence
        """
        if not self.camera_enabled or self.latest_camera_frame is None:
            return None, None, 0
        
        depth = self.latest_camera_frame["depth"]
        if depth is None:
            return None, None, 0
        
        height, width = depth.shape
        
        # Convert PyBullet depth buffer to actual distance
        # PyBullet returns linearized depth: depth_buffer = far * near / (far - (far - near) * depth)
        # We need to convert back to actual distance
        far = self.camera_far
        near = self.camera_near
        # Avoid division by zero
        depth_linear = np.clip(depth, 0.001, 0.999)
        actual_depth = far * near / (far - (far - near) * depth_linear)
        
        # Sample multiple horizontal strips for robustness
        strip_rows = [height // 3, height // 2, 2 * height // 3]
        
        best_opening = None
        best_score = 0
        
        for row in strip_rows:
            strip = actual_depth[row, :]
            
            # Find regions where depth exceeds wall threshold (openings)
            is_opening = strip > wall_threshold
            
            # Find contiguous opening regions
            openings = []
            start = None
            for i, is_open in enumerate(is_opening):
                if is_open and start is None:
                    start = i
                elif not is_open and start is not None:
                    if i - start >= min_opening_width:
                        avg_depth = np.mean(strip[start:i])
                        openings.append((start, i, avg_depth))
                    start = None
            # Don't forget last region
            if start is not None and width - start >= min_opening_width:
                avg_depth = np.mean(strip[start:width])
                openings.append((start, width, avg_depth))
            
            # Score each opening: prefer centered, wide, and deep openings
            for start_col, end_col, avg_depth in openings:
                center_col = (start_col + end_col) / 2
                opening_width = end_col - start_col
                
                # Center score: prefer openings near center of view
                center_offset = abs(center_col - width / 2) / (width / 2)
                center_score = 1 - center_offset
                
                # Width score: wider is better
                width_score = min(opening_width / 50, 1.0)
                
                # Depth score: deeper (farther away can still see through) is better
                depth_score = min(avg_depth / 5.0, 1.0)
                
                # Combined score
                score = center_score * 0.3 + width_score * 0.4 + depth_score * 0.3
                
                if score > best_score:
                    best_score = score
                    best_opening = (center_col, opening_width, avg_depth)
        
        if best_opening is None:
            step = len(self.xy_coordinates) - 1 if self.xy_coordinates else 0
            if step % 500 == 0:
                # Log depth stats for debugging when no door found
                strip_row = height // 2
                strip = actual_depth[strip_row, :]
                print(f"[DOOR-DEBUG] Step {step}: No door found. Depth stats: "
                      f"min={np.min(strip):.2f}, max={np.max(strip):.2f}, "
                      f"median={np.median(strip):.2f}, threshold={wall_threshold:.1f}")
            return None, None, 0
        
        center_col, opening_width, avg_depth = best_opening
        
        # Convert pixel column to angle relative to robot heading
        # FOV is camera_fov degrees, centered at width/2
        fov_rad = np.radians(self.camera_fov)
        # Pixel offset from center, normalized to -1 to 1
        pixel_offset = (center_col - width / 2) / (width / 2)
        # Convert to angle (negative because image is mirrored in some setups)
        door_angle = -pixel_offset * (fov_rad / 2)
        
        # Confidence based on score
        confidence = best_score
        
        # Debug logging
        step = len(self.xy_coordinates) - 1 if self.xy_coordinates else 0
        if step % 200 == 0:
            pos = self.xy_coordinates[-1] if self.xy_coordinates else [0, 0]
            heading_deg = np.degrees(self.orientation_angle[-1]) if self.orientation_angle else 0
            print(f"[DOOR-DEBUG] Step {step}: pos=({pos[0]:.1f},{pos[1]:.1f}), heading={heading_deg:.0f}°, "
                  f"door at col={center_col:.0f}/{width} (angle={np.degrees(door_angle):.1f}°), "
                  f"width={opening_width}px, depth={avg_depth:.1f}m, conf={confidence:.2f}")
        
        return door_angle, avg_depth, confidence

    def compute_movement(self, gc_network, pc_network, cognitive_map, exploration_phase=True):
        """Compute and set motor gains of agents. Simulate the movement with py-bullet"""

        gains = self.avoid_obstacles(gc_network, pc_network, cognitive_map, exploration_phase)

        self.change_speed(gains)
        p.stepSimulation()

        self.save_position_and_speed()
        if self.camera_enabled:
            self.capture_camera_frame()
        if self.visualize:
            time.sleep(self.dt/5)

    def change_speed(self, gains):
        p.setJointMotorControlArray(bodyUniqueId=self.carID,
                                    jointIndices=[4, 6],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=gains,
                                    forces=[10, 10])

    def save_position_and_speed(self):
        [position, angle] = p.getBasePositionAndOrientation(self.carID)
        angle = p.getEulerFromQuaternion(angle)
        
        # NaN protection: if position becomes invalid, try to recover using last known position
        if np.any(np.isnan(position)) or np.any(np.isnan(angle)):
            if len(self.xy_coordinates) > 0:
                # Use last known good position
                print(f"[WARNING] Robot position/angle is NaN, using last known position")
                self.xy_coordinates.append(self.xy_coordinates[-1])
                self.orientation_angle.append(self.orientation_angle[-1])
                self.xy_speeds.append([0.0, 0.0])
                self.speeds.append(0.0)
                return
            else:
                # No previous position, use origin
                print(f"[WARNING] Robot position is NaN and no previous position available")
                self.xy_coordinates.append(np.array([0.0, 0.0]))
                self.orientation_angle.append(0.0)
                self.xy_speeds.append([0.0, 0.0])
                self.speeds.append(0.0)
                return
        
        self.xy_coordinates.append(np.array([position[0], position[1]]))
        self.orientation_angle.append(angle[2])

        [linear_v, _] = p.getBaseVelocity(self.carID)
        self.xy_speeds.append([linear_v[0], linear_v[1]])
        self.speeds.append(np.linalg.norm([linear_v[0], linear_v[1]]))

    def compute_gains(self):
        """Calculates motor gains based on heading and goal vector direction"""
        current_angle = self.orientation_angle[-1]
        
        # NaN protection: if angle or goal_vector is NaN, return zero gains
        if np.isnan(current_angle) or np.any(np.isnan(self.goal_vector)):
            return [0.0, 0.0]
        
        current_heading = [np.cos(current_angle), np.sin(current_angle)]
        diff_angle = compute_angle(current_heading, self.goal_vector) / np.pi
        
        # NaN protection for diff_angle (can happen if goal_vector is zero)
        if np.isnan(diff_angle):
            diff_angle = 0.0

        gain = min(np.linalg.norm(self.goal_vector) * 5, 1)

        # If close to the goal do not move
        if gain < 0.5:
            gain = 0

        # If large difference in heading, do an actual turn
        if abs(diff_angle) > 0.05 and gain > 0:
            max_speed = self.max_speed / 2
            direction = np.sign(diff_angle)
            if direction > 0:
                v_left = max_speed * gain * -1
                v_right = max_speed * gain
            else:
                v_left = max_speed * gain
                v_right = max_speed * gain * -1
        else:
            # Otherwise only adjust course slightly
            self.turning = False
            max_speed = self.max_speed
            v_left = max_speed * (1 - diff_angle * 2) * gain
            v_right = max_speed * (1 + diff_angle * 2) * gain

        # Diagnostic: log motor state every 500 steps when in upper area
        step = len(self.xy_coordinates)
        if step % 500 == 0 and len(self.xy_coordinates) > 0 and self.xy_coordinates[-1][1] > 8.0:
            pos = self.xy_coordinates[-1]
            spd = np.linalg.norm(self.xy_speeds[-1]) if self.xy_speeds else 0
            print(f"[MOTOR] Step {step}: pos=({pos[0]:.1f},{pos[1]:.1f}), gv=({self.goal_vector[0]:.2f},{self.goal_vector[1]:.2f}), "
                  f"norm={np.linalg.norm(self.goal_vector):.2f}, gain={gain:.2f}, diff_a={diff_angle:.3f}, "
                  f"motors=[{v_left:.1f},{v_right:.1f}], speed={spd:.3f}")

        return [v_left, v_right]

    def _line_of_sight_clear(self, from_pos, to_pos, z=0.1):
        """True if a straight line from from_pos to to_pos hits no wall.

        Uses PyBullet's rayTest to detect any rigid body along the segment.
        We exclude the agent's own body (``self.carID``) from the hit set so
        the test only reports environment obstacles (walls, blocks, etc.).
        """
        start = [float(from_pos[0]), float(from_pos[1]), z]
        end = [float(to_pos[0]),   float(to_pos[1]),   z]
        hits = p.rayTest(start, end)
        if not hits:
            return True
        for hit in hits:
            hit_uid = hit[0]
            if hit_uid != -1 and hit_uid != self.carID:
                return False
        return True

    def retreat_to_safe_position(self, max_lookback=200):
        """Retreat to the last position where the agent was moving.
        Biologically: a rat backing out of a dead-end passage.

        Bounded by ``max_lookback`` trajectory entries so that retreat can
        only undo local oscillation — not catastrophically unwind progress
        like a previously-completed wall crossing. Each candidate retreat
        position is additionally line-of-sight checked against the current
        position via raycasting; candidates separated from the agent by a
        wall are rejected. This prevents the resetBasePositionAndOrientation
        teleport from phasing the agent through closed doors when the
        trajectory history contains a coordinate on the far side of a wall
        (which can happen if PyBullet's contact resolver briefly lets the
        agent's centre overshoot a thin collider during wall oscillation).
        """
        current_pos = np.array(self.xy_coordinates[-1])
        retreat_pos = None

        lookback_start = max(0, len(self.xy_coordinates) - 1 - max_lookback)
        for i in range(len(self.xy_coordinates) - 1, lookback_start - 1, -1):
            candidate = np.array(self.xy_coordinates[i])
            if np.linalg.norm(candidate - current_pos) <= 0.5:
                continue
            if not self._line_of_sight_clear(current_pos, candidate):
                # A wall lies between the agent and this candidate; teleport
                # would phase through it. Keep searching further back.
                continue
            retreat_pos = candidate
            break

        if retreat_pos is None:
            # Fall back to a short step opposite the current heading, but
            # still verify line-of-sight; if that's also blocked, abort the
            # retreat rather than teleport through geometry.
            heading = self.orientation_angle[-1] if self.orientation_angle else 0.0
            fallback = current_pos - np.array([np.cos(heading), np.sin(heading)]) * 0.5
            if self._line_of_sight_clear(current_pos, fallback):
                retreat_pos = fallback
            else:
                print(f"[RETREAT] Aborted at ({current_pos[0]:.1f},{current_pos[1]:.1f}): "
                      f"no line-of-sight retreat candidate within {max_lookback} steps")
                return

        print(f"[RETREAT] From ({current_pos[0]:.1f},{current_pos[1]:.1f}) "
              f"to ({retreat_pos[0]:.1f},{retreat_pos[1]:.1f})")

        current_orn = p.getBasePositionAndOrientation(self.carID)[1]
        p.resetBasePositionAndOrientation(
            self.carID, [retreat_pos[0], retreat_pos[1], 0.1], current_orn)
        p.resetBaseVelocity(self.carID, [0, 0, 0], [0, 0, 0])

        self.xy_coordinates.append(retreat_pos.copy())
        self.orientation_angle.append(self.orientation_angle[-1])
        self.xy_speeds.append([0.0, 0.0])
        self.speeds.append(0.0)

    def end_simulation(self):
        p.disconnect()

    def avoid_obstacles(self, gc_network, pc_network, cognitive_map, exploration_phase):
        """Main controller function, to check for obstacles and adjust course if needed."""
        # During exploration (initial), skip expensive navigation computations
        # Just do basic obstacle avoidance
        if exploration_phase:
            ray_reference = p.getLinkState(self.carID, 0)[1]
            current_heading = p.getEulerFromQuaternion(ray_reference)[2]
            goal_vector_angle = np.arctan2(self.goal_vector[1], self.goal_vector[0])
            
            scan_angles = np.linspace(0, 2 * np.pi, num=8, endpoint=False)
            scan_dist = self.ray_detection(scan_angles)
            minimum_dist = np.min(scan_dist)
            
            if minimum_dist < 0.3:
                # Back up from obstacle
                idx = np.argmin(scan_dist)
                angle = scan_angles[idx] + np.pi
                self.goal_vector = np.array([np.cos(angle), np.sin(angle)]) * 0.5
                self.goal_vector_original = self.goal_vector
            
            return self.compute_gains()
        
        # Normal navigation mode - full obstacle avoidance
        ray_reference = p.getLinkState(self.carID, 0)[1]
        current_heading = p.getEulerFromQuaternion(ray_reference)[2]  # in radians
        goal_vector_angle = np.arctan2(self.goal_vector[1], self.goal_vector[0])
        angles = np.linspace(0, 2 * np.pi, num=self.num_ray_dir, endpoint=False)

        # direction where we want to check for obstacles
        angles = np.append(angles, [goal_vector_angle, current_heading])

        ray_dist = self.ray_detection(angles)  # determine ray values
        self._last_min_ray_dist = float(np.min(ray_dist[:self.num_ray_dir]))  # cache for metrics
        changed = self.update_directions(ray_dist)  # check if an direction became unblocked

        # Stay in topology-based mode even when all directions are free.
        # Switching to perform_look_ahead_2x (vector-based) doesn't see walls
        # and produces bad vectors in walled environments.

        minimum_dist = np.min(ray_dist)
        if minimum_dist < 0.3:
            # Initiate back up maneuver
            idx = np.argmin(ray_dist)
            angle = angles[idx] + np.pi
            self.goal_vector = np.array([np.cos(angle), np.sin(angle)]) * 0.5
            self.goal_vector_original = self.goal_vector
            self.topology_based = True

        if not exploration_phase:
            if self.topology_based or ray_dist[-1] < 0.6 or ray_dist[-2] < 0.6:
                # Approaching an obstacle in heading or goal vector direction, or topology based
                if not self.topology_based or changed:
                    # Switching to topology-based, or already topology-based but new direction became available
                    self.topology_based = True
                    pick_intermediate_goal_vector(gc_network, pc_network, cognitive_map, self)

        return self.compute_gains()

    def update_directions(self, ray_dist):
        """Check which of the directions are blocked and if one became unblocked"""
        changed = False
        directions = np.ones_like(self.directions)
        for idx in range(self.num_ray_dir):
            left = idx - 1 if idx - 1 >= 0 else self.num_ray_dir - 1  # determine left direction in circle
            right = idx + 1 if idx + 1 <= self.num_ray_dir - 1 else 0  # determine right direction in circle
            if ray_dist[idx] < 1.3 or ray_dist[left] < 0.9 or ray_dist[right] < 0.9:
                # If in direction an obstacle is nearby or in one of the neighbouring, then it is blocked
                directions[idx] = False
            if idx % self.num_travel_dir == 0 and directions[idx] and not self.directions[idx]:
                # One of the traveling directions became unblocked
                changed = True
        self.directions = directions
        return changed

    def ray_detection(self, angles):
        """Check for obstacles in defined directions."""

        p.removeAllUserDebugItems()

        ray_len = 2  # max_ray length to check for

        ray_from = []
        ray_to = []

        ray_from_point = np.array(p.getLinkState(self.carID, 0)[0])
        ray_from_point[2] = ray_from_point[2] + 0.02

        for angle in angles:
            ray_from.append(ray_from_point)
            ray_to.append(np.array([
                np.cos(angle) * ray_len + ray_from_point[0],
                np.sin(angle) * ray_len + ray_from_point[1],
                ray_from_point[2]
            ]))

        ray_dist = np.empty_like(angles)
        results = p.rayTestBatch(ray_from, ray_to, numThreads=0)
        for idx, result in enumerate(results):
            hit_object_uid = result[0]

            dist = ray_len
            if hit_object_uid != -1:
                hit_position = result[3]
                dist = np.linalg.norm(hit_position - ray_from_point)
                ray_dist[idx] = dist

            ray_dist[idx] = dist

            # if dist < 1:
            #     p.addUserDebugLine(ray_from[idx], ray_to[idx], (1, 1, 1))

        return ray_dist




