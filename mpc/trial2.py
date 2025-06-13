import numpy as np
import casadi as ca
from pathlib import Path
from urdfpy import URDF
import pybullet as p
import pybullet_data
import csv
import sys
import time

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





# Setup controllable joints
controllable_joints = [1, 2]
# for j in range(p.getNumJoints(robot_id)):
#     info = p.getJointInfo(robot_id, j)
#     if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
#         controllable_joints.append(j)

NUM_TRAJECTORIES = 50

# Load robot model from URDF file
robot = URDF.load(URDF_FILE)

# Identify the indexes for the links and joints from the URDF
link1 = next(link for link in robot.links if link.name == "link1")
link2 = next(link for link in robot.links if link.name == "link2")
joint1 = next(j for j in robot.joints if j.name == "joint1")
joint2 = next(j for j in robot.joints if j.name == "joint2")


# Constants parsing from the URDF
L_C1 = link1.inertial.origin[:3, 3][0] # Location of center of mass, link 1
L_C2 = link2.inertial.origin[:3, 3][0] # Location of center of mass, link 2
I1 = link1.inertial.inertia[1, 1] # Iyy, link 1
I2 = link2.inertial.inertia[1, 1] # Iyy, link 1
M1 = link1.inertial.mass # Mass, link 1
M2 = link2.inertial.mass # Mass, link 2
L1 = link1.visuals[0].geometry.box.size[0] # Length, link 1
L2 = link2.visuals[0].geometry.box.size[0] # Length, link 2
D1 = joint1.dynamics.damping if joint1.dynamics and joint1.dynamics.damping is not None else 0.0 # Damping coefficient, joint 1
D2 = joint2.dynamics.damping if joint2.dynamics and joint2.dynamics.damping is not None else 0.0 # Damping coefficient, joint 2

def compute_joint_torques_from_ee_force(q, fx, fy):
    """
    Computes the joint torques required to apply a desired force at the end-effector.

    Args:
        q: List or array of joint angles [q1, q2]
        fx: Force in the X direction at the end-effector
        fy: Force in the Y direction at the end-effector

    Returns:
        tau: np.array of joint torques [tau1, tau2]
    """
    q1, q2 = q

    # Compute the Jacobian matrix (2x2)
    J = np.array([
        [-L1*np.sin(q1) - L2*np.sin(q1 + q2), -L2*np.sin(q1 + q2)],
        [ L1*np.cos(q1) + L2*np.cos(q1 + q2),  L2*np.cos(q1 + q2)]
    ])

    # End-effector force vector as column
    F_ee = np.array([[fx], [fy]])  # shape (2,1)

    # Compute joint torques: tau = J^T * F_ee
    tau = (J.T @ F_ee).flatten()   # shape (2,)

    return tau

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

def _get_joint_torques(robot_id):
    torques = []
    joint_states = p.getJointStates(robot_id, controllable_joints)
    reaction_torques = [state[3] for state in joint_states]
    torques.append(reaction_torques)
    return torques

def _get_joint_positions(robot_id):
    positions = []
    joint_states = p.getJointStates(robot_id, controllable_joints)
    position = [state[0] for state in joint_states] 
    positions.append(position)
    return positions

def _get_joint_velocities(robot_id):
    velocities = []
    joint_states = p.getJointStates(robot_id, controllable_joints)
    velocity = [state[1] for state in joint_states] 
    velocities.append(velocity)
    return velocities

def _get_joint_reactions(robot_id):
    accelerations = []
    joint_states = p.getJointStates(robot_id, controllable_joints)
    acceleration = [state[2] for state in joint_states] 
    accelerations.append(acceleration)
    return accelerations

def _get_all_joint_reactions(robot_id):
    reactions = []
    joint_states = p.getJointStates(robot_id, [0, 1, 2, 3])
    reaction = [state[2] for state in joint_states] 
    reactions.append(reaction)
    return reactions

def compute_static_torques(joint_positions, joint_velocities, joint_accelerations, external_force, sim_steps=100):
    print(f"joint_positions: {joint_positions}")
    print(f"joint_velocities: {joint_velocities}")
    print(f"joint_accelerations: {joint_accelerations}")


        # Start physics server
    # Start PyBullet
    DT = 1/250 # Time step constant
    g = 0 # Gravity acceleration
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -g)
    p.setTimeStep(DT)
    # Load URDF in pybullet
    robot_id = p.loadURDF(URDF_FILE, basePosition=(0, 0, 0.07), #  0.07 space from the floor to prevent collisions.
                        baseOrientation=p.getQuaternionFromEuler((0, 3.14, 0)), # Rotated 180° to point upwards.
                        useFixedBase=True) # Fixed on the base

    # Reset joint states (positions and velocities)
    for idx, joint_index in enumerate(controllable_joints):
        p.resetJointState(robot_id, joint_index, joint_positions[idx], joint_velocities[idx])
    p.stepSimulation()
    print(f"torques: {_get_joint_torques(robot_id)}")
    
    p.enableJointForceTorqueSensor(robot_id, jointIndex=0, enableSensor=True)
    p.enableJointForceTorqueSensor(robot_id, jointIndex=1, enableSensor=True)
    p.enableJointForceTorqueSensor(robot_id, jointIndex=2, enableSensor=True)
    p.enableJointForceTorqueSensor(robot_id, jointIndex=3, enableSensor=True)


    p.setJointMotorControlArray(robot_id, jointIndices=controllable_joints, controlMode=p.TORQUE_CONTROL, targetPositions=joint_positions, forces=[0] * len(controllable_joints))
    p.stepSimulation()
    print(f"torques: {_get_joint_torques(robot_id)}")

    # test = p.calculateInverseDynamics(robot_id, joint_positions, joint_velocities, [0.0] * len(controllable_joints))
    reactions = _get_all_joint_reactions(robot_id)  # (Fx, Fy, Fz, Mx, My, Mz)

    print("Joint 0 reaction force/torque:", reactions[0][0])
    print("Joint 1 reaction force/torque:", reactions[0][1])
    print("Joint 2 reaction force/torque:", reactions[0][2])
    print("Joint 3 reaction force/torque:", reactions[0][3])

    torques = []
    for _ in range(15):
        world_pos = p.getLinkState(robot_id, 3)[0]
            # Apply external force at the end-effector
        p.applyExternalForce(robot_id, 3, forceObj=external_force,
                         posObj=world_pos, flags=p.WORLD_FRAME)
        p.stepSimulation()
        torques.append(_get_joint_torques(robot_id))
    
    measured_torque = torques[-1]
    print(f"torques: {measured_torque}")
    pos = _get_joint_positions(robot_id)[0]
    vel = _get_joint_velocities(robot_id)[0]

    C, M, G, D = compute_dynamics(pos[0], pos[1], joint_velocities[0], joint_velocities[1], -g)
    print(f"Matrix M: {M}")
    C_np = np.array(C.full()).flatten()
    M_numeric = np.array(M.full())
    M_acc = M_numeric @ joint_accelerations
    G_np = np.array(G.full()).flatten()
    D_np = np.array(D.full()).flatten()

    difference = C_np + M_acc + G_np + D_np

    reactions = _get_all_joint_reactions(robot_id)  # (Fx, Fy, Fz, Mx, My, Mz)

    print("Joint 0 reaction force/torque:", reactions[0][0])
    print("Joint 1 reaction force/torque:", reactions[0][1])
    print("Joint 2 reaction force/torque:", reactions[0][2])
    print("Joint 3 reaction force/torque:", reactions[0][3])

    print(f" The measured torque is {measured_torque}")
    print(f" Component of mass {M_acc}")
    print(f" Component of Coriolis {C_np}")
    print(f" Component of gravity {G_np}")
    print(f" Component of damping {D_np}")

    print(f" The difference is : {difference}")

    # # Calculate inverse dynamics using pybullet
    # inv_dyn_torque = p.calculateInverseDynamics(robot_id, joint_positions, joint_velocities, joint_accelerations, physicsClientId=client)
    # print(f"PyBullet inverse dynamics torque: {inv_dyn_torque}")
    result = difference-measured_torque
    p.disconnect()
    return result

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


def _complete_force():
    # Inputs
    state = ca.MX.sym("state", 6)   # [theta1, theta2, theta1_dot, theta2_dot]
    torque = ca.MX.sym("torque", 2)
    g = 9.81

    # Unpack state
    theta1 = state[0]
    theta2 = state[1]
    theta1_dot = state[2]
    theta2_dot = state[3]
    accel = ca.vertcat(state[4], state[5])

    # --- Jacobian ---
    s1 = ca.sin(theta1)
    c1 = ca.cos(theta1)
    s12 = ca.sin(theta1 + theta2)
    c12 = ca.cos(theta1 + theta2)

    J = ca.MX.zeros(2, 2)
    J[0, 0] = -L1 * s1 - L2 * s12
    J[0, 1] = -L2 * s12
    J[1, 0] = L1 * c1 + L2 * c12
    J[1, 1] = L2 * c12
    J[1, :] *= -1  # flip y row

    # --- Dynamics components ---
    M11 = I1 + I2 + M1 * L_C1**2 + M2 * (L1**2 + L_C2**2 + 2 * L1 * L_C2 * ca.cos(theta2))
    M12 = I2 + M2 * (L_C2**2 + L1 * L_C2 * ca.cos(theta2))
    M22 = I2 + M2 * L_C2**2
    M = ca.vertcat(ca.horzcat(M11, M12), ca.horzcat(M12, M22))

    C = ca.vertcat(
        -M2 * L1 * L2 * ca.sin(theta2) * (2 * theta1_dot * theta2_dot + theta2_dot**2),
        M2 * L1 * L2 * ca.sin(theta2) * theta1_dot**2
    )

    G = ca.vertcat(
        -g * (M1 * L_C1 * ca.cos(theta1) + M2 * L1 * ca.cos(theta1) + M2 * L_C2 * ca.cos(theta1 + theta2)),
        -M2 * g * L_C2 * ca.cos(theta1 + theta2)
    )

    D = ca.vertcat(-D1 * theta1_dot, -D2 * theta2_dot)

    # --- External torque and force ---
    tau_dyn = M @ accel + C + G + D
    tau_ext = tau_dyn - torque


    # Use direct inverse of J^T
    F = ca.pinv(J.T) @ tau_ext

    return ca.Function("force_from_state_and_torque", [state, torque], [F])


def _compute_dynamics():

    theta1 = ca.MX.sym("theta1")
    theta2 = ca.MX.sym("theta2")
    theta1_dot = ca.MX.sym("theta1_dot")
    theta2_dot = ca.MX.sym("theta2_dot")
    g= ca.MX.sym("g")

    # Mass-inertia matrix
    M11 = I1 + I2 + M1 * L_C1**2 + M2 * (L1**2 + L_C2**2 + 2 * L1 * L_C2 * ca.cos(theta2))
    M12 = I2 + M2 * (L_C2**2 + L1 * L_C2 * ca.cos(theta2))
    M22 = I2 + M2 * L_C2**2
    M = ca.vertcat(ca.horzcat(M11, M12), ca.horzcat(M12, M22))

    # Coriolis and centrifugal forces
    C = ca.vertcat(
        -M2 * L1 * L2 * ca.sin(theta2) * (2 * theta1_dot * theta2_dot + theta2_dot**2),
        M2 * L1 * L2 * ca.sin(theta2) * theta1_dot**2)
        # Gravity effects
    G = ca.vertcat(
        -g * (M1 * L_C1 * ca.cos(theta1) + M2 * L1 * ca.cos(theta1) + M2 * L_C2 * ca.cos(theta1 + theta2)),
        -M2 * g * L_C2 * ca.cos(theta1 + theta2))
    
        # Damping force
    F_damping = ca.vertcat(-D1 * theta1_dot, -D2 * theta2_dot)
    
    return ca.Function("force_from_state_and_torque", [theta1, theta2, theta1_dot, theta2_dot, g], [C, M, G, F_damping])
    


if __name__ == "__main__":

    states = load_data(TRAJECTORIES, NUM_TRAJECTORIES, 499)
    actions = load_data(ACTIONS, NUM_TRAJECTORIES, 499-1)
    complete_force = _complete_force()
    force_jacobian = _force_jacobian()
    compute_dynamics = _compute_dynamics()

    for traj_idx in range(states.shape[0]):
        # print(f"The initial states are: {states[traj_idx, 0]}")
        for step in range(actions.shape[1]):
            try:
                state = states[traj_idx, step+1]
                torque = actions[traj_idx, step]    

                # METHOD 1: Compute force from state and torque using all dynamics
                F_e = complete_force(state, torque)
                F_e_np = np.array(F_e.full()).flatten() 
                print(f"The computed force1 is: {F_e_np}")
                
                # METHOD 2: Compute force from state and torque using only the Jacobian
                F_e_2 = force_jacobian(state[0], state[1], torque)
                F_e_2_np = np.array(F_e_2.full()).flatten()
                print(f"The computed force2 is: {F_e_2_np}")

                # Results
                print(f"Given torque: {torque}.")
                # VALIDATION METHOD 1: Compute torque from force using the Jacobian
                val_torque = compute_joint_torques_from_ee_force(state[:2], F_e_np[0], F_e_np[1])

                # VALIDATION METHOD 1: Compute torque from force using the Jacobian
                val_torque2 = compute_joint_torques_from_ee_force(state[:2], F_e_2_np[0], F_e_2_np[1])

                # # VALIDATION METHOD 2: Compute torque from force using the pybullet simulation
                # val_torque3 = compute_static_torques(state[:2], state[2:4], state[4:6], [F_e_np[0],0,F_e_np[1]], sim_steps=5)

                # VALIDATION METHOD 2: Compute torque from force using the pybullet simulation
                val_torque4 = compute_static_torques(state[:2], state[2:4], state[4:6], [F_e_2_np[0],0,F_e_2_np[1]], sim_steps=5)
                
 
                print(f"Computed torque 1.1: {val_torque}. Computed torque 1.2: {val_torque2}.")
                # print(f"Computed torque 2.1: {-1*val_torque3}. Computed torque 2.2: {val_torque4}.")


            except Exception as e:
                print(f"Error at trajectory {traj_idx}, step {step}: {e}")
                continue
    p.disconnect()