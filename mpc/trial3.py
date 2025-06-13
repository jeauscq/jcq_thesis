import pybullet as p
import csv
import numpy as np
import pybullet_data
import casadi as ca
import os
import sys
from pathlib import Path

# ——— Project Modules ————————————————————————————————————————
current_file = Path().resolve()
while current_file.name != "jcq_thesis" and current_file != current_file.parent:
    current_file = current_file.parent
sys.path.append(str(current_file))

from utils.config import BASE_DIR


# BASE_DIR = "/home/jeauscq/Desktop/jcq_thesis/"
DATASETS_DIR = BASE_DIR + "datasets/"
URDF_FILE = BASE_DIR + '/models/2dof_planar_robot.urdf'
TRAJECTORIES = DATASETS_DIR + "Policy/MPC/torque_Const/mpc_generated_tor_constraint_dataset_acc.csv"
ACTIONS =  DATASETS_DIR + "Policy/MPC/torque_Const/mpc_generated_tor_constraint_actions_fixed.csv"

NUM_TRAJECTORIES = 25
TRAJECTORY_LEN = 498

DT = 1/250 # Time step constant
g = -9.81 # Gravity acceleration
controllable_joints = [1, 2] # Joint indices of the robot

client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, g)
p.setTimeStep(DT)
# Load URDF in pybullet
robot_id = p.loadURDF(URDF_FILE, basePosition=(0, 0, 0.07), #  0.07 space from the floor to prevent collisions.
baseOrientation=p.getQuaternionFromEuler((0, 3.14, 0)), # Rotated 180° to point upwards.
useFixedBase=True) # Fixed on the base



def load_data(path, num_trajectories, expected_len):
    data = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_trajectories:
                break
            parts = line.strip().split(",")
            traj = [list(map(float, entry.split(";"))) for entry in parts]
            assert len(traj) == expected_len, f"Trajectory length mismatch in line {i}, obtained {len(traj)}, expected {expected_len}"
            data.append(np.array(traj))
    return np.array(data)


def _force_jacobian():
    theta1 = ca.MX.sym("theta1")
    theta2 = ca.MX.sym("theta2")
    l1 = 1.0
    l2 = 1.0
    s1 = ca.sin(theta1)
    c1 = ca.cos(theta1)
    s12 = ca.sin(theta1 + theta2)
    c12 = ca.cos(theta1 + theta2)

    J = ca.MX.zeros(2, 2)
    J[0, 0] = -l1 * s1 - l2 * s12
    J[0, 1] = -l2 * s12
    J[1, 0] = l1 * c1 + l2 * c12
    J[1, 1] = l2 * c12


    # J[1, :] *= -1

    tau = ca.MX.sym("tau", 2, 1)

    # Use direct inverse of J^T
    F = ca.pinv(J.T) @ tau

    return ca.Function("F_e_with_dynamics", [theta1, theta2, tau], [F])


if __name__ == "__main__":

    states = load_data(TRAJECTORIES, NUM_TRAJECTORIES, TRAJECTORY_LEN + 1)
    actions = load_data(ACTIONS, NUM_TRAJECTORIES, TRAJECTORY_LEN)
    force_jacobian = _force_jacobian()

with open("get_ee_dynamic_componentes_dataset_final.csv", "w", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["traj_idx", "step", "force_norm", "torque_norm", "jacobian_condition"])

    for traj_idx in range(states.shape[0]):
        print(f"The initial states are: {states[traj_idx, 0]}")
        for i, joint in enumerate(controllable_joints):
            p.resetJointState(robot_id, joint, states[traj_idx, 0, i], states[traj_idx, 0, 2+i])
        for step in range(actions.shape[1]):
            state = states[traj_idx, step+1]
            torque = actions[traj_idx, step]    
            theta1, theta2 = state[0], state[1]
            dtheta1, dtheta2 = state[2], state[3]


            # Reset joint states (positions and velocities)
            for idx, joint_index in enumerate(controllable_joints):
                p.resetJointState(robot_id, joint_index, state[idx], state[idx + 2])
            p.setJointMotorControlArray(robot_id, jointIndices=controllable_joints, controlMode=p.TORQUE_CONTROL, forces=torque)
            p.stepSimulation()

            new_joint_states = p.getJointStates(robot_id, controllable_joints)
            new_positions = [s[0] for s in new_joint_states]
            new_velocities = [s[1] for s in new_joint_states]
            new_accelerations = [s[2] for s in new_joint_states]  # These are usually zeros unless simulation computes them

            F = force_jacobian(new_positions[0], new_positions[1], torque)
            print(f"F: {F}")

            # Compute inverse dynamics torques
            num_joints = p.getNumJoints(robot_id)

            positions = [p.getJointState(robot_id, j)[0] for j in range(num_joints)]
            velocities = [p.getJointState(robot_id, j)[1] for j in range(num_joints)]
            accelerations = [0.0] * num_joints

            assert all(np.isfinite(positions)), "Non-finite positions"
            assert all(np.isfinite(velocities)), "Non-finite velocities"
            assert all(np.isfinite(accelerations)), "Non-finite accelerations"

            inv_dyn = p.calculateInverseDynamics(robot_id, positions, velocities, accelerations)

            print(f"inv_dyn: {inv_dyn}")
