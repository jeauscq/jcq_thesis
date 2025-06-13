import pinocchio as pin
import pybullet as p
import pybullet_data
import numpy as np
import time
from scipy.integrate import solve_ivp

# Load Pinocchio URDF model
model = pin.buildModelFromUrdf("/home/jeauscq/Desktop/jcq-thesis/models/2dof_planar_robot.urdf")
data = model.createData()

dt = 0.003  # Integration time step (3ms)

# Define initial state and torque input #####################################
state = np.array([1.1750, 0.2134, 1.125, 3])  # [q1, q2, dq1, dq2]
tau = np.array([1.0, 0.5])  # Torque inputs

def dynamics(t, state, tau):
    """ Computes the state derivative using Pinocchio """
    q_0 = state[:2]  # Joint positions
    qdot = state[2:]  # Joint velocities

    M = pin.crba(model, data, q_0)
    C = pin.computeCoriolisMatrix(model, data, q_0, qdot)
    G = pin.computeGeneralizedGravity(model, data, q_0)
    # print(M, C, G)
    qddot = np.linalg.solve(M, tau - C @ qdot - G)

    return np.hstack((qdot, qddot))  # Returns [dq, dqdot]


def integrate_solve_ivp(state, tau):
    """ Uses solve_ivp to integrate the motion over dt """
    start_time = time.time()
    sol = solve_ivp(dynamics, [0, dt], state, args=(tau,), method='RK45', t_eval=[dt])
    elapsed_time = time.time() - start_time
    return sol.y.flatten(), elapsed_time


def dynamics_solve(state, tau):
    """ Computes next state using single step integration """
    start_time = time.time()
    dx = dynamics(0, state, tau)
    elapsed_time = time.time() - start_time
    return state + dx * dt, elapsed_time  # Euler step (not highly accurate)


def get_controllable_joints(robot_id):
    # Identify controllable joints (revolute/prismatic) in the PyBullet model.
    controllable_joints = []
    num_joints = p.getNumJoints(robot_id)

    for j in range(num_joints):
        joint_info = p.getJointInfo(robot_id, j)
        joint_type = joint_info[2]  # Joint type

        # PyBullet joint types: 0 = revolute, 1 = prismatic
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            controllable_joints.append(j)

    return controllable_joints

def disable_uncontrolled_joints(robot_id):
    # Disable velocity and position control for non-controllable joints.
    num_joints = p.getNumJoints(robot_id)

    for j in range(num_joints):
        if j not in controllable_joints:
            p.setJointMotorControl2(robot_id, j, controlMode=p.VELOCITY_CONTROL, force=0)

# Pinocchio - Single Step

state_euler, elapsed_time_euler = dynamics_solve(state, tau)


# Pinocchio - solve_ivp
state_ivp, elapsed_time_ivp = integrate_solve_ivp(state, tau)

# PyBullet Simulation Setup
p.connect(p.DIRECT)  # Use p.GUI for visualization
robot_id = p.loadURDF("/home/jeauscq/Desktop/jcq-thesis/models/2dof_planar_robot.urdf", useFixedBase=True)
controllable_joints = get_controllable_joints(robot_id)  # Get controllable joints

# Set the initial joint positions
for i, q in enumerate(state[:2]):
    p.resetJointState(robot_id, controllable_joints[i], q, targetVelocity=state[i+2])

# Disables position/velocity control on the joints
p.setJointMotorControlArray(robot_id, jointIndices=controllable_joints, controlMode=p.VELOCITY_CONTROL,
                            forces=[0] * len(controllable_joints))

# Apply torque control
p.setJointMotorControlArray(
    robot_id,
    jointIndices=controllable_joints,
    controlMode=p.TORQUE_CONTROL,
    forces=tau
)

# Step the simulation forward by 3ms
p.setTimeStep(dt)
start_time = time.time()
p.stepSimulation()

# Get new joint states from PyBullet
pybullet_states = []
for i in controllable_joints:  
    joint_state = p.getJointState(robot_id, i)
    pybullet_states.append(joint_state[0])  # Position
pybullet_states = np.array(pybullet_states)

elapsed_time_pybullet = time.time() - start_time

# Print results
print("\nUsing Pinocchio (Euler Step):", state_euler[:2])
print("Execution time (s):", elapsed_time_euler)

print("\nUsing Pinocchio (solve_ivp):", state_ivp[:2])
print("Execution time (s):", elapsed_time_ivp)

print("\nUsing PyBullet:")
print("Predicted joint positions:", pybullet_states)
print("Execution time (s):", elapsed_time_pybullet)

# Print difference
print("\nDifference (Euler - PyBullet):", state_euler[:2] - pybullet_states)
print("Difference (solve_ivp - PyBullet):", state_ivp[:2] - pybullet_states)
