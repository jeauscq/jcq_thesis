import os.path as osp
import pybullet as p
import random
import math
import numpy as np
import sys
import pybullet_data
import csv

URDF_PATH = "/home/jeauscq/Desktop/jcq_thesis/models/2dof_planar_robot.urdf"
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../pybullet_ompl/'))
import pb_ompl

class TrajectoryGenerator():
    def __init__(self, steps, dt, compute_accelerations, gui=True):
        self.obstacles = []
        self.steps = steps
        self.dt = dt
        self.gui = gui
        self.compute_accelerations = compute_accelerations

        if self.gui:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(3.5, 40, -40, [0, 0, 0])
        else:
            p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./500)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = p.loadURDF("plane.urdf")
        self.obstacles.append(plane)

        robot_id = p.loadURDF(
            URDF_PATH,
            basePosition=(0, 0, 0.05),
            baseOrientation=p.getQuaternionFromEuler((0, 3.14, 0)),
            useFixedBase=1
        )

        robot = pb_ompl.PbOMPLRobot(robot_id)
        self.robot = robot
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
        self.pb_ompl_interface.set_planner("BITstar")

    def compute_joint_derivates(self, position_list):
        position_array = np.array(position_list)
        num_steps = position_array.shape[0]
        velocity_array = np.zeros_like(position_array)
        for i in range(1, num_steps):
            velocity_array[i] = (position_array[i] - position_array[i - 1]) / self.dt
        if self.compute_accelerations:
            acceleration_array = np.zeros_like(position_array)
            for i in range(1, num_steps):
                acceleration_array[i] = (velocity_array[i] - velocity_array[i - 1]) / self.dt
            final_list = np.hstack((position_array, velocity_array, acceleration_array)).tolist()
        else:
            final_list = np.hstack((position_array, velocity_array)).tolist()
        return final_list

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)

    def disconnect_pybullet(self):
        p.disconnect()

    def add_obstacles(self, start, goal, start_coor, goal_coor):
        box_size = [0.1, 0.1, 0.1]
        x = random.uniform(start_coor[0], goal_coor[0])
        z = random.uniform(start_coor[2], goal_coor[2])
        box_position = [x, 0, z]
        self.box_position = box_position
        self.add_box(box_position, box_size)
        self.pb_ompl_interface.set_obstacles(self.obstacles)

        attempts = 0
        while not (self.is_state_valid(start) and self.is_state_valid(goal)) and attempts < 15:
            attempts += 1
            self.clear_obstacles()
            x = random.uniform(start_coor[0], goal_coor[0])
            z = random.uniform(start_coor[2], goal_coor[2])
            box_position = [x, 0, z]
            self.box_position = box_position
            self.add_box(box_position, box_size)
            self.pb_ompl_interface.set_obstacles(self.obstacles)

    def add_box(self, box_pos, box_size):
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[i/2 for i in box_size])
        box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos)
        self.obstacles.append(box_id)

    def is_state_valid(self, state):
        return self.pb_ompl_interface.is_state_valid(state)

    def generate_random_valid_position(self):
        start, goal = None, None
        while start is None:
            start_candidate = np.random.uniform(-np.pi, np.pi, size=2).tolist()
            if self.is_state_valid(start_candidate):
                start = start_candidate
        while goal is None:
            goal_candidate = np.random.uniform(-np.pi, np.pi, size=2).tolist()
            if self.is_state_valid(goal_candidate):
                goal = goal_candidate
        return start, goal

    def forward_kinematics(self, theta, L1=1, L2=1):
        theta[0] = math.pi - theta[0]
        theta[1] = -theta[1]
        x = -(L1 * np.cos(theta[0]) + L2 * np.cos(theta[0] + theta[1]))
        y = 0
        z = L1 * np.sin(theta[0]) + L2 * np.sin(theta[0] + theta[1])
        return [x, y, z]

    def add_debug_sphere(self, position, type=' ', radius=0.05):
        if type == 'start':
            rgbaColor = [0, 1, 0, 0.6]
        elif type == 'goal':
            rgbaColor = [0, 0, 1, 0.6]
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

    def main(self):
        start, goal = self.generate_random_valid_position()
        start_coor = self.forward_kinematics(start)
        goal_coor = self.forward_kinematics(goal)
        self.add_obstacles(start, goal, start_coor, goal_coor)
        box_position = self.box_position

        self.robot.set_state(start)

        if self.gui:
            self.add_debug_sphere(position=start_coor, type='start')
            self.add_debug_sphere(position=goal_coor, type='goal')

        res, path = self.pb_ompl_interface.plan(goal)

        states = []
        if res:
            states = self.compute_joint_derivates(path)
        self.disconnect_pybullet()
        return res, states, box_position, start_coor, goal_coor

def save_trajectory_csv(new_data, file_path):
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        row = [";".join(map(str, state)) for state in new_data]
        writer.writerow(row)

def save_positions_csv(pos_data, pos_file_path):
    with open(pos_file_path, "a", newline="") as f:
        writer = csv.writer(f)
        row = ["{:.5f},{:.5f},{:.5f}".format(*entry) for entry in pos_data]
        writer.writerow(row)

def read_trajectories_csv(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            trajectory = [list(map(float, state.split(";"))) for state in row]
            yield trajectory

if __name__ == '__main__':
    file_path = "../new_training_trajectories.csv"
    pos_file_path = file_path.replace(".csv", "_positions.csv")

    valid_num = 0
    while valid_num < 1382:
        env = TrajectoryGenerator(gui=False, steps=500, dt=1/250, compute_accelerations=False)
        valid, states, obstacle_pos, start_coor, goal_coor = env.main()
        if valid:
            valid_num += 1
            save_trajectory_csv(states, file_path)
            save_positions_csv([obstacle_pos, start_coor, goal_coor], pos_file_path)

    for trajectory in read_trajectories_csv(file_path):
        print(trajectory[0])