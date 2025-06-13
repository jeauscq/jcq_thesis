import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path
import random
import csv
import torch

URDF_PATH = str(Path(__file__).resolve().parents[2])+ "/models/2dof_planar_robot.urdf"
BASE_DIR = "/home/jeauscq/Desktop/jcq_thesis/"
POLICY_TRAJECTORIES_NORM_STATS = BASE_DIR + "datasets/training_trajectories_policy_n_stats.txt"

class Custom2DoFEnv(gym.Env):
    """
    Custom Gymnasium environment for a 2-DoF torque-controlled planar robot in PyBullet.
    Observations consist of N past states, the current state, and M future desired states.
    The robot must track a desired trajectory with torque actions applied to the joints.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}


    def __init__(self, trajectories, render_mode=None, max_steps=380, N=5, M=5, discriminator=None, usePositions=False, normalize_path=None):
        """
        Initializes the environment.

        Args:
            render_mode (str): Mode to render the environment ('human' for GUI).
            max_steps (int): Maximum number of steps per episode.
            N (int): Number of past states to include in the observation.
            M (int): Number of future desired states to include in the observation.
        """
        self.urdf_path = URDF_PATH
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
        self.trajectories = trajectories

        self.normalize = normalize_path if normalize_path is not None else POLICY_TRAJECTORIES_NORM_STATS
        self.mean, self.std = self._load_stats(POLICY_TRAJECTORIES_NORM_STATS)

        self.usePositions = usePositions
        self.positions = self._load_positions() if self.usePositions else []
        self.obstacle_pos = []
        self.box_id = None

        # Pybullet conf
        self.client_id = p.connect(p.GUI if render_mode == "human" else p.DIRECT)  # Connect to physics client
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Needed to later load the plane
        p.setTimeStep(self.time_step)  # Set simulation time step
        p.setGravity(0, 0, -9.81)  # Set gravity

        # Define observation and action spaces
        obs_len = (self.N + 1 + self.M) * self.state_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)
        self.single_observation_space = self.observation_space
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
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
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
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
        if self.usePositions:
            self.obstacle_pos = self.positions[index][0]
            # print(index, self.obstacle_pos)
            # Now here the obstacle is loaded in the environment
            box_size = [0.1, 0.1, 0.1]
            half_extents = [dim / 2 for dim in box_size]
            colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visBoxId = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1, 0, 0, 1])  # Red box

            self.box_id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex=colBoxId,baseVisualShapeIndex=visBoxId,
                                            basePosition=self.obstacle_pos, physicsClientId=self.client_id)

        self.traj_step = random.randint(10, 200)

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

        # Reward based on tracking error
        desired_state = np.array(self.trajectory[self.traj_step])     # shape: (4,)
        actual_state = np.array(self.current_state)                   # shape: (4,)
        error = actual_state - desired_state

        # Separate pos and vel
        pos_error = error[:2]
        vel_error = error[2:]

        # Obstacle penalization
        obs_penalty = 0

        if self.usePositions:
            # Penalization for getting close to the obstacles
            threshold = 0.15
            closest_points = p.getClosestPoints(bodyA=self.robot_id, bodyB=self.box_id, 
                                                distance=threshold, physicsClientId=self.client_id)
            # If no points returned, robot is further than threshold
            if len(closest_points) > 0:
                min_distance = min(cp[8] for cp in closest_points)  # cp[8] is contactDistance
            else:
                min_distance = float('inf')
            # The closer the robot is to the obstacle, the higher the penalty
            # The penalty is defined as the inverse of the distance to the obstacle
            obs_penalty = 1.0 / (min_distance + 1e-5)  # small epsilon to avoid division by zero

        alpha = 0.5   # weight for position error
        beta = 0.5 # weight for velocity error
        gamma = 0.0 # weight for obstacle proximity

        reward = -10* (alpha * np.sum(pos_error**2) + beta * np.sum(vel_error**2) + gamma * obs_penalty)/(alpha + beta + gamma)

        self.total_reward += reward
        info = {}

        # Check robot-ground contacts
        contact_points = p.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id, physicsClientId=self.client_id)

        if self.usePositions:
            # Check robot-box contacts
            box_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.box_id, physicsClientId=self.client_id)
            contact_points = contact_points + box_contacts
        
        # Check if any collisions exist
        has_collision = len(contact_points) > 0

        # Combine termination conditions
        terminated = has_collision # or terminate_on_tracking_error

        if has_collision:
            info['early_termination_reason'] = 'robot_ground_collision_ex_base'


        truncated = self.step_counter >= self.max_steps or self.traj_step + self.M >= len(self.trajectory)-1

        return obs, reward, terminated, truncated, info


    def render(self):
        """
        Renders the simulation with a top-down camera view if render_mode is 'human'.

        Returns:
            px (np.ndarray): RGB frame.
        """
        if self.render_mode == "human":
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.5],
                distance=1.2,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)
            _, _, px, _, _ = p.getCameraImage(320, 240, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
            return px


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


    def _load_positions(self):
        """
        Loads the positions file. 
        They are saved as "obsx, obsy, obsz","startx, starty, startz","endx, endy, endz5"
        """
        with open(POSITIONS_FILE, 'r') as file:
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
