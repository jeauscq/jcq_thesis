"""
This script compares the symbolic dynamics of a 2DOF planar robot with the PyBullet simulation. 
It computes the next state using both methods and compares the results.
The script uses PyBullet to simulate the robot's dynamics and CasADi to compute the symbolic dynamics.
The script also computes the mass matrix, Coriolis and centrifugal forces, and gravitational forces using PyBullet.
The script is designed to be run in a standalone mode, and it can be easily modified to test different initial states and control actions.
"""

import pybullet as p
import pybullet_data
import numpy as np
from pathlib import Path
import time
import sys

# ——— Project Modules ————————————————————————————————————————
current_file = Path().resolve()
while current_file.name != "jcq_thesis" and current_file != current_file.parent:
    current_file = current_file.parent
sys.path.append(str(current_file))

from mpc.casadi_mpc import simulate_next_state

# ——— Constants ————————————————————————————————————————
USE_GUI = False
TIME_STEP = 1.0 / 250.0
BASE_DIR = "/home/jeauscq/Desktop/jcq_thesis/"
URDF_PATH = BASE_DIR + "/models/2dof_planar_robot.urdf"

# Hardcoded test case
initial_state = np.array([0.5, 1.2, 0.3, -0.1])  # [θ1, θ2, θ1_dot, θ2_dot]
# initial_state = np.random.uniform(size = 4, low = -np.pi, high = np.pi)   # [θ1, θ2, θ1_dot, θ2_dot]

control_action = np.array([5.0, -5.0])  # torque input

# Symbolic simulation
symbolic_result = simulate_next_state(initial_state, control_action, dt=TIME_STEP)

symbolic_result = np.array(symbolic_result).flatten()

# Start PyBullet
client = p.connect(p.GUI if USE_GUI else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)
p.setTimeStep(TIME_STEP)
p.loadURDF("plane.urdf")

robot_id = p.loadURDF(URDF_PATH, basePosition=(0, 0, 0.07),
                      baseOrientation=p.getQuaternionFromEuler((0, 3.14, 0)),
                      useFixedBase=True)
# Disable all damping
for j in range(p.getNumJoints(robot_id)):
    p.changeDynamics(robot_id, j, lateralFriction=0.0, linearDamping=0.0, angularDamping=0.0)

# Base too
p.changeDynamics(robot_id, -1, lateralFriction=0.0, linearDamping=0.0, angularDamping=0.0)

# Setup joints
controllable_joints = []
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        controllable_joints.append(j)

q = [0.5, 1.2]
q_dot = [0.3, -0.1]
q_ddot = [0.0, 0.0]  # For isolating C + G
tau = control_action
# Get mass matrix
M_pb = p.calculateMassMatrix(robot_id, q)
tau_CG = p.calculateInverseDynamics(robot_id, q, q_dot, [0.0, 0.0])
tau_G = p.calculateInverseDynamics(robot_id, q, [0.0, 0.0], [0.0, 0.0])
tau_C = np.array(tau_CG) - np.array(tau_G)
print("---PyBullet Dynamics---")
print("G =", tau_G)
print("C =", tau_C)
print("M =", M_pb)

# Get C + G from inverse dynamics
c_plus_g_pb = p.calculateInverseDynamics(robot_id, q, q_dot, [0.0, 0.0])

# Solve for q_ddot using PyBullet's own model
tau = control_action
q_ddot_pb_model = np.linalg.solve(np.array(M_pb), np.array(tau) - np.array(c_plus_g_pb))

print("q_ddot (PyBullet model):", q_ddot_pb_model)

# Set initial state
p.resetJointState(robot_id, controllable_joints[0], initial_state[0], initial_state[2])
p.resetJointState(robot_id, controllable_joints[1], initial_state[1], initial_state[3])

# Get velocities before stepping
vel_before = [p.getJointState(robot_id, j)[1] for j in controllable_joints]

p.setJointMotorControlArray(robot_id, controllable_joints, controlMode=p.VELOCITY_CONTROL, forces=[0, 0])
p.setJointMotorControlArray(robot_id, controllable_joints, controlMode=p.TORQUE_CONTROL, forces=[0, 0])

# Apply one step of control
p.setJointMotorControlArray(robot_id, controllable_joints, controlMode=p.TORQUE_CONTROL, forces=control_action)
p.stepSimulation()
# Get velocities after stepping
vel_after = [p.getJointState(robot_id, j)[1] for j in controllable_joints]

# Compute acceleration
q_ddot_pybullet = (np.array(vel_after) - np.array(vel_before)) / TIME_STEP
print("Estimated q_ddot (PyBullet):", q_ddot_pybullet)

time.sleep(5 if USE_GUI else 0)

# Read result
pb_pos = [p.getJointState(robot_id, j)[0] for j in controllable_joints]
pb_vel = [p.getJointState(robot_id, j)[1] for j in controllable_joints]
pybullet_result = np.array(pb_pos + pb_vel)

# Comparison
print("\n--- Single Step Comparison ---")
print("Initial State:         ", initial_state)
print("Control Action:        ", control_action)
print("Symbolic Next State:   ", symbolic_result)
print("PyBullet Next State:   ", pybullet_result)
print("Absolute Error:        ", np.abs(symbolic_result - pybullet_result))

p.disconnect()
