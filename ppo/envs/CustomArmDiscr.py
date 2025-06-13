import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path
import random
import math
import csv
import time
import sys
import torch

# ——— Project Modules ————————————————————————————————————————
current_file = Path().resolve()
while current_file.name != "jcq_thesis" and current_file != current_file.parent:
    current_file = current_file.parent
sys.path.append(str(current_file))

from utils.config import BASE_DIR

URDF_PATH = BASE_DIR + "/models/2dof_planar_robot.urdf"
PLANE_PATH = BASE_DIR + "/models/clean_plane.urdf"

class Custom2DoFEnv(gym.Env):
    """
    Custom Gymnasium environment for a 2-DoF torque-controlled planar robot in PyBullet.
    Observations consist of N past states, the current state, and M future desired states.
    The robot must track a desired trajectory with torque actions applied to the joints.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}


    def __init__(self, trajectories, render_mode=None, max_steps=380, N=5, M=5, discriminator=None, positions=None, normalize_path=None, rewardMode=0, fps=60, idx=0):
        """
        Initializes the environment.

        Args:
            render_mode (str): Mode to render the environment ('human' for GUI).
            max_steps (int): Maximum number of steps per episode.
            N (int): Number of past states to include in the observation.
            M (int): Number of future desired states to include in the observation.
        """
        self.urdf_path = URDF_PATH
        self.plane_path = PLANE_PATH
        self.discriminator = discriminator
        self.device = torch.device("cuda" if (self.discriminator != None and torch.cuda.is_available()) else "cpu")
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.N = int(N)
        self.M = int(M)
        self.step_counter = 0
        self.state_dim = 4  # [joint1_pos, joint2_pos, joint1_vel, joint2_vel]
        self.trajectory = None
        self.traj_step = 0
        self.time_step = 1.0 / 250.0
        self.fps = fps
        self.trajectories = trajectories
        self.rewardMode = rewardMode

        self.normalize = normalize_path
        self.mean, self.std = self._load_stats(self.normalize)

        self.positions = self._load_positions(positions) if positions!=None else []
        self.idx = idx if positions is not None else 0
        self.obstacle_pos = []
        self.box_id = None

        self.render_mode = render_mode

        # Pybullet conf
        self.client_id = p.connect(p.GUI if self.render_mode == "man" else p.DIRECT)  # Connect to physics client
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Needed to later load the plane
        p.setTimeStep(self.time_step)  # Set simulation time step
        p.setGravity(0, 0, -9.81)  # Set gravity

        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Hide the side menus
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # 
            
            p.resetDebugVisualizerCamera(
                cameraDistance=1.8,     # Closer distance to the scene
                cameraYaw=0,           # Horizontal rotation
                cameraPitch=2,        # Vertical angle (negative looks down)
                cameraTargetPosition=[0, 0, 0.75]  # Point to look at (usually robot base)
            )
        # Define observation and action spaces
        obs_len = (self.N + 1 + self.M) * self.state_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)
        self.single_observation_space = self.observation_space

        if self.rewardMode != 0:
            self.u_min = np.array([-38, -20], dtype=np.float32)
            self.u_max = np.array([33, 18], dtype=np.float32)
        if self.rewardMode <2:
            self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        else:            
            # Define limits for clipping for modes 2 and 3
            self.action_space = spaces.Box(low=np.array(self.u_min, dtype=np.float32), high=np.array(self.u_max, dtype=np.float32), dtype=np.float32)
        self.single_action_space = self.action_space

        # Initializes the current and past states
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.past_buffer = [np.zeros(self.state_dim, dtype=np.float32) for _ in range(self.N)]
        self.robot_id = None


    def reset(self, seed=None, options=None):
        """
        Resets the environment state for a new episode.

        Returns:
            obs (np.ndarray): Initial observation.
            info (dict): Extra diagnostic information.
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.step_counter = 0
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)

        # Load environment and robot
        self.plane_id = p.loadURDF(self.plane_path, useFixedBase=True ,physicsClientId=self.client_id)
        self.robot_id = p.loadURDF(self.urdf_path, basePosition=(0, 0, 0.07),
                                   baseOrientation=p.getQuaternionFromEuler((0, 3.14, 0)),
                                   useFixedBase=1,
                                   physicsClientId=self.client_id)
        
        # Defines the controllable joints of the robot
        self._get_controllable_joints()
        assert len(self.controllable_joints) == 2

        # Disable default motor control
        p.setJointMotorControlArray(self.robot_id, jointIndices=self.controllable_joints, controlMode=p.VELOCITY_CONTROL, 
                                    forces=[0] * len(self.controllable_joints), physicsClientId=self.client_id)
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=self.controllable_joints,
                                    controlMode=p.TORQUE_CONTROL,forces=[0] * len(self.controllable_joints),
                                    physicsClientId=self.client_id)
        # Load a random trajectory
        index = random.randint(0, len(self.trajectories) - 1)
        self.trajectory = self.trajectories[index]
            
        # An obstacle is added
        if self.positions != []:
            self.obstacle_pos = self.positions[self.idx][0]
            # print(index, self.obstacle_pos)
            # Now here the obstacle is loaded in the environment
            box_size = [0.06, 0.06, 0.06]
            half_extents = [dim / 2 for dim in box_size]
            colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visBoxId = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1, 0, 0, 1])  # Red box

            self.box_id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex=colBoxId,baseVisualShapeIndex=visBoxId,
                                            basePosition=self.obstacle_pos, physicsClientId=self.client_id)
        

        self.traj_step = random.randint(10, 200)
        # self.max_steps = len(self.trajectory) - self.traj_step - self.M - 1

        if self.render_mode == "human":
            # Calcular coordenadas en espacio cartesiano para los puntos de inicio y fin
            start_pos = self.forward_kinematics(self.trajectory[self.traj_step][:2])
            end_pos = self.forward_kinematics(self.trajectory[min(self.traj_step + self.max_steps, len(self.trajectory)-1 -self.M)][:2])
            self._add_debug_sphere(start_pos, type='start', radius=0.05)
            self._add_debug_sphere(end_pos, type='goal', radius=0.05)
            self.capture_steps = sorted(random.sample(range(self.max_steps), 10))  # store 10 steps to capture
            self.image_counter = 0  # initialize image file counter


        # Load past buffer and current state
        self.past_buffer = self.trajectory[self.traj_step - self.N:self.traj_step]
        self.current_state = self.trajectory[self.traj_step]

        # Set initial joint positions
        iter_size = int(self.state_dim / len(self.controllable_joints))
        for i, joint in enumerate(self.controllable_joints):
                p.resetJointState(int(self.robot_id), int(joint), float(self.current_state[i]), targetVelocity=float(self.current_state[int(i+iter_size)]), physicsClientId=int(self.client_id))

        self.total_reward = 0.0
        return self._get_obs(), {}


    def _get_obs(self):
        """
        Builds the observation from past, current, and future states. It normalizes the observation using the mean and std if specified.

        Returns:
            obs (np.ndarray): Flattened observation vector.
        """
        past_states = np.array(self.past_buffer).flatten()
        current_state = np.array(self.current_state).flatten()
        future_states = np.array(self.trajectory[self.traj_step + 1 : self.traj_step + 1 + self.M]).flatten()
        obs = np.concatenate([past_states, current_state, future_states]).astype(np.float32)

        obs = obs.reshape(-1, 4)
        obs = (obs - self.mean) / self.std # Normalizes the observations based on the stats of the reference dataset
        obs = obs.flatten().astype(np.float32)
        return obs


    def forward_kinematics(self, theta, L1=1, L2=1):
        theta[0] = math.pi - theta[0]
        theta[1] = -theta[1]
        x = -(L1 * np.cos(theta[0]) + L2 * np.cos(theta[0] + theta[1]))
        y = 0
        z = L1 * np.sin(theta[0]) + L2 * np.sin(theta[0] + theta[1])
        return [-x, y, z]
    

    def step(self, action):
        """
        Applies action and advances the simulation by one time step.

        Args:
            action (np.ndarray): Torque values for the joints.

        Returns:
            obs (np.ndarray): New observation.
            reward (float): Reward for the step.
            terminated (bool): Whether episode ended due to task failure.
            truncated (bool): Whether episode was truncated by max_steps or data end.
            info (dict): Extra diagnostic information.
        """
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=self.controllable_joints,
                                    controlMode=p.TORQUE_CONTROL, forces=action.tolist(), physicsClientId=self.client_id)
        p.stepSimulation(physicsClientId=self.client_id)
        self.step_counter += 1
        self.traj_step += 1

        # Update past buffer
        if len(self.past_buffer)>0:
            self.past_buffer[:-1] = self.past_buffer[1:]
            self.past_buffer[-1] = self.current_state
    

        # Update current state
        new_state = []
        for joint in self.controllable_joints:
            joint_state = p.getJointState(self.robot_id, joint, physicsClientId=self.client_id)
            new_state.append(joint_state[0])  # Position
        for joint in self.controllable_joints:
            joint_state = p.getJointState(self.robot_id, joint, physicsClientId=self.client_id)
            new_state.append(joint_state[1])  # Velocity
        self.current_state = new_state

        obs = self._get_obs()

        # Prepare structured observation
        obs_window = obs.reshape(self.N + 1 + self.M, self.state_dim)
        obs_tensor = torch.tensor(obs_window, dtype=torch.float32).unsqueeze(0).to(self.device)

        if self.discriminator is not None:
            with torch.no_grad():
                self.discriminator.eval()
                reward_outputs = self.discriminator(obs_tensor)  # <- this is a list of tensors
                reward_tensor = torch.stack(reward_outputs).clamp(-1, 1).mean()
                reward = reward_tensor.item()
            if self.rewardMode in [1, 2]:
                clip_threshold = 1.5
                safe_boundary = 3
                if action[0]>self.u_min[0]+safe_boundary and action[0]<self.u_max[0]-safe_boundary and action[1]>self.u_min[1]+safe_boundary and action[1]<self.u_max[1]-safe_boundary:
                    reward_constraints = 0
                else:
                    u_norm = 2 * (action - self.u_min) / (self.u_max - self.u_min) - 1 # Normalizes the control actions to make sure that the closer the control action to the bounds, the worst evaluated it is
                    u_norm_clipped = np.clip(u_norm, -clip_threshold, clip_threshold)
                    reward_constraints = -0.1 * np.mean(u_norm_clipped**2) # The worst case scenario is a penalty of -0.225    
                reward += reward_constraints
        else:
            # In case no reward is needed
            reward = 0.0       
        self.total_reward += reward
        info = {}
        # List of link indices to monitor for collisions (assuming links 1 and 2 are indices 1 and 2)
        relevant_links = [1, 2]

        # Check robot-ground contacts (filtering to links 1 and 2)
        contact_points = [
            cp for cp in p.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id, physicsClientId=self.client_id)
            if cp[3] in relevant_links
        ]

        if self.positions:
            # Check robot-box contacts (also filtering to links 1 and 2)
            box_contacts = [
                cp for cp in p.getContactPoints(bodyA=self.robot_id, bodyB=self.box_id, physicsClientId=self.client_id)
                if cp[3] in relevant_links
            ]

            contact_points += box_contacts
        # Check if any collisions exist
        has_collision = len(contact_points) > 0

        # Combine termination conditions
        terminated = has_collision # or terminate_on_tracking_error

        if has_collision:
            if self.step_counter == 1:
                info['early_termination_reason'] = 'unfeasible_initial_state'
                # print("Collision detected at step 1, terminating episode.")
            elif self.positions and len(box_contacts) > 0:
                info['early_termination_reason'] = 'robot_box_collision'
            else:
                info['early_termination_reason'] = 'robot_ground_collision_ex_base'


        truncated = self.step_counter >= self.max_steps or self.traj_step >= len(self.trajectory)-1 -self.M

        if self.render_mode == "human":
                # time.sleep(1 / self.fps)
                if self.step_counter%30==0: 
                    width, height = 1920 * 2, 1080 * 2  # 4K render

                    _, _, px, _, _ = p.getCameraImage(
                        width, height,
                        viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                            cameraTargetPosition=[0, 0, 1],
                            distance=2.8,
                            yaw=0,
                            pitch=10,
                            roll=0,
                            upAxisIndex=2
                        ),
                        projectionMatrix=p.computeProjectionMatrixFOV(
                            fov=60,
                            aspect=width / height,
                            nearVal=0.1,
                            farVal=100.0
                        ),
                        renderer=p.ER_BULLET_HARDWARE_OPENGL,
                        lightDirection=[1, 1, 1],
                        shadow=True,
                        lightColor=[1, 1, 1]
                    )

                    from PIL import Image

                    rgba = np.reshape(px, (height, width, 4)).astype(np.uint8)
                    rgb = rgba[:, :, :3]

                    # Replace white background with light blue (RGB = [255, 255, 255] → [180, 190, 230])
                    mask = np.all(rgb == [255, 255, 255], axis=-1)
                    rgb[mask] = [180, 190, 230]

                    img = Image.fromarray(rgb)
                    img.save(f"Images/mod2/render_snapshot_traj{self.idx}_{self.image_counter:02d}.png", dpi=(300, 300))
                    self.image_counter += 1

        return obs, reward, terminated, truncated, info


    def render(self):
        """
        Renders the simulation with a top-down camera view if render_mode is 'human'.

        Returns:
            px (np.ndarray): RGB frame.
        """
        if self.render_mode == "human":
            view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0.75],
                distance=1.8,
                yaw=0,
                pitch=-1,
                roll=0,
                upAxisIndex=2)
            proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)
            _, _, px, _, _ = p.getCameraImage(320, 240, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
            return px

    def _add_debug_sphere(self, position, type=' ', radius=0.05):
        if type == 'start':
            rgbaColor = [0, 1, 0, 1]
        elif type == 'goal':
            rgbaColor = [0, 0, 1, 1]
        else:
            rgbaColor = [1, 1, 1]
        sphere_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgbaColor
        )
        sphere_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=sphere_visual,
            baseCollisionShapeIndex=-1,
            basePosition=position
        )
        return sphere_id

    def _get_controllable_joints(self):
        """
        Identifies controllable joints (revolute or prismatic) in the robot.
        Sets self.controllable_joints.
        """
        controllable_joints = []
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        for j in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)
            joint_type = joint_info[2]
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                controllable_joints.append(j)
        self.controllable_joints = controllable_joints


    def _load_positions(self, positions_file):
        """
        Loads the positions file. 
        They are saved as "obsx, obsy, obsz","startx, starty, startz","endx, endy, endz5"
        """
        with open(positions_file, 'r') as file:
            reader = csv.reader(file)
            data = [row for row in reader]

        # Convert each "x,y,z" string to list of floats
        return  [[list(map(float, item.split(','))) for item in row] for row in data]

    
    def _load_stats(self, file_path):
        mean = []
        std = []
        with open(file_path, 'r') as f:
            for line in f:
                if 'mean' in line:
                    mean = [float(x) for x in line.strip().split(':')[1].strip().split()]
                elif 'std' in line:
                    std = [float(x) for x in line.strip().split(':')[1].strip().split()]
        return np.array(mean), np.array(std)

    def close(self):
        """
        Disconnects from PyBullet simulation.
        """
        p.disconnect(self.client_id)
