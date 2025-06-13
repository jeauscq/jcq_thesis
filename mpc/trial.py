# The formulas used to symbolically model the 2DoF robot come from [A. A. Okubanjo, et al., “Modeling of 2 dof robot arm and control,” Futo Journal Series, vol. 3, no. 2, 2017]

import casadi as ca
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
import numpy as np
import sys
from pathlib import Path
from urdfpy import URDF
import csv


# ——— Project Modules ————————————————————————————————————————
current_file = Path().resolve()
while current_file.name != "jcq_thesis" and current_file != current_file.parent:
    current_file = current_file.parent
sys.path.append(str(current_file))

from utils.config import BASE_DIR

# BASE_DIR = "/home/jeauscq/Desktop/jcq_thesis/"
DATASETS_DIR = BASE_DIR + "datasets/"
URDF_FILE = BASE_DIR + '/models/2dof_planar_robot_with_tool.urdf'
TRAJECTORIES = DATASETS_DIR + "Policy/MPC/torque_Const/mpc_generated_tor_constraint_dataset_acc.csv"
ACTIONS =  DATASETS_DIR + "Policy/MPC/torque_Const/mpc_generated_tor_constraint_actions_fixed.csv"

# === CONFIG ===
NUM_TRAJECTORIES = 150
TRAJECTORY_LEN = 498  # 498 control actions, 499 states
STATE_DIM = 4
ACTION_DIM = 2

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

# MPC implementation
DT = 1/250 # Time step constant
g = 9.81  # Gravity acceleration


# Start PyBullet
DT = 1/250 # Time step constant
g = 9.81  # Gravity acceleration
client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -g)
p.setTimeStep(DT)
# Load URDF in pybullet
robot_id = p.loadURDF(URDF_FILE, basePosition=(0, 0, 0.07), #  0.07 space from the floor to prevent collisions.
                      baseOrientation=p.getQuaternionFromEuler((0, 3.14, 0)), # Rotated 180° to point upwards.
                      useFixedBase=True) # Fixed on the base
ee_joint = 3
p.enableJointForceTorqueSensor(robot_id, ee_joint)

# Setup controllable joints
controllable_joints = []
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        controllable_joints.append(j)

# Disable velocity control and enables torque control
p.setJointMotorControlArray(robot_id, controllable_joints, controlMode=p.VELOCITY_CONTROL, forces=[0, 0])
p.setJointMotorControlArray(robot_id, controllable_joints, controlMode=p.TORQUE_CONTROL, forces=[0, 0])

def get_mass_matrix_pybullet(q):
    return np.array(p.calculateMassMatrix(robot_id, q))

def get_gravity_vector_pybullet(q):
    # set velocities and accelerations to zero
    zero_vel = [0.0] * len(controllable_joints)
    zero_acc = [0.0] * len(controllable_joints)
    return np.array(p.calculateInverseDynamics(robot_id, q, zero_vel, zero_acc))


def get_coriolis_vector_pybullet(q, q_dot):
    zero_acc = [0.0] * len(controllable_joints)
    tau_full = np.array(p.calculateInverseDynamics(robot_id, q, q_dot, zero_acc))
    tau_gravity = get_gravity_vector_pybullet(q)
    coriolis = tau_full - tau_gravity
    return coriolis

def get_jacobian_pybullet(q, q_dot, torque):
    zero_acc = [0.0] * len(controllable_joints)
    
    p.setJointMotorControlArray(bodyUniqueId=robot_id, jointIndices=controllable_joints, controlMode=p.TORQUE_CONTROL, forces=torque.tolist())    

    # print(f"End effector index: {p.getNumJoints(robot_id) - 1 }")
    # p.applyExternalForce(objectUniqueId=robot_id, linkIndex=ee_link_index, forceObj=[20, 0, 0], posObj=[0, 0, 0], flags=p.LINK_FRAME)
    
    # p.stepSimulation()

    # Step 1: Get joint torques
    tau = torque

    # # Step 2: Get joint positions for Jacobian
    # q = [p.getJointState(robot_id, j)[0] for j in controllable_joints]
    # qd = [p.getJointState(robot_id, j)[1] for j in controllable_joints]
    # qdd = [0.0] * len(controllable_joints)

    # # Step 3: Compute Jacobian
    # J_lin, J_ang = p.calculateJacobian(
    #     robot_id, ee_joint, [0, 0, 0], q, qd, qdd
    # )
    # J = np.vstack((np.array(J_lin), np.array(J_ang)))  # 6xN

    # # Step 4: Compute wrench
    # tau = np.array(tau).reshape((-1, 1))  # Nx1
    # wrench_tcp = np.linalg.pinv(J.T) @ tau  # 6x1
    # print("Estimated force robot would apply at TCP if blocked:\n", wrench_tcp)

    joint_state = p.getJointState(robot_id, ee_joint)

    wrench = joint_state[2]  # [Fx, Fy, Fz, Mx, My, Mz]
    print("End-effector Wrench (in sensor frame):", wrench)

    print(f"Torques in joint 2. Written {tau[1]} vs perceived {p.getJointState(robot_id, 2)[2]}")
    # J_lin, J_ang = p.calculateJacobian(robot_id, 2, [1, 0, 0], q, q_dot, zero_acc)
    # J = np.array(J_lin)[[0,2], :]  # Extract 2D task space (x, z)
    # return J

def build_force_function_unregularized():
    J = ca.MX.sym("J", 2, 2)
    tau = ca.MX.sym("tau", 2, 1)
    F = ca.solve(J.T, tau)
    return ca.Function("F_e_unreg", [J, tau], [F])

def complete_force():
    # Inputs
    state = ca.MX.sym("state", 6)   # [theta1, theta2, theta1_dot, theta2_dot]
    torque = ca.MX.sym("torque", 2)

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

def get_dynamics_components():
    theta1 = ca.MX.sym("theta1")
    theta2 = ca.MX.sym("theta2")
    theta1_dot = ca.MX.sym("theta1_dot")
    theta2_dot = ca.MX.sym("theta2_dot")
    
    # External forces (for now set to 0)
    F = ca.vertcat(0, 0)

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

    return ca.Function("dynamics_components", [theta1,theta2,theta1_dot,theta2_dot], [M, C, G, F_damping])

def get_ee_dynamic_components():
    theta1 = ca.MX.sym("theta1")
    theta2 = ca.MX.sym("theta2")
    theta1_dot = ca.MX.sym("theta1_dot")
    theta2_dot = ca.MX.sym("theta2_dot")
    torque = ca.MX.sym("torque", 2)
    
    # External forces (for now set to 0)
    F = ca.vertcat(0, 0)

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
    J[1, :] *= -1

    J_dot = ca.MX.zeros(2,2)
    J_dot[0, 0] = -l1*c1*theta1_dot-l2*c12*(theta1_dot+theta2_dot)
    J_dot[0, 1] = -l2*c12*(theta1_dot+theta2_dot)
    J_dot[1, 0] = -l1*s1*theta1_dot-l2*s12*(theta1_dot+theta2_dot)
    J_dot[1, 1] = -l2*s12*(theta1_dot+theta2_dot)

    we_d = J @ ca.inv(M) @ (torque - C - G - F_damping) + J_dot @ ca.vertcat(theta1_dot, theta2_dot)

    lambda_e = ca.inv(J @ ca.inv(M @ J.T)) 
    mu_e = lambda_e @ J @ca.inv(M) @ C - lambda_e @ J_dot @ ca.vertcat(theta1_dot, theta2_dot) 
    p_e = lambda_e @ J @ ca.inv(M) @ G 

    F_e = lambda_e @ we_d + mu_e + p_e

    print("F_e", F_e)

    return ca.Function("dynamics_components", [theta1,theta2,theta1_dot,theta2_dot, torque], [F_e, J, M, C, G, F_damping])

def build_force_function_reg():
    J = ca.MX.sym("J", 2, 2)
    tau = ca.MX.sym("tau", 2, 1)
    C = ca.MX.sym("C", 2, 1)
    G = ca.MX.sym("G", 2, 1)
    D = ca.MX.sym("D", 2, 1)

    tau_dyn = C + G + D
    tau_ext = tau - tau_dyn

    # Use direct inverse of J^T
    F = ca.pinv(J.T) @ tau_ext

    # Use direct inverse of J^T
    # F = ca.solve(J.T, tau_ext)

    return ca.Function("F_e_with_dynamics", [J, tau, C, G, D], [F])

def build_force_function_unreg():
    J = ca.MX.sym("J", 2, 2)
    tau = ca.MX.sym("tau", 2, 1)
    C = ca.MX.sym("C", 2, 1)
    G = ca.MX.sym("G", 2, 1)
    D = ca.MX.sym("D", 2, 1)

    tau_dyn = C + G + D
    tau_ext = tau - tau_dyn

    # Use direct inverse of J^T
    F = ca.solve(J.T, tau_ext)

    return ca.Function("F_e_with_dynamics_unreg", [J, tau, C, G, D], [F])

# Helper to compute CasADi-based force
def build_jacobian_casadi_function():
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

    J[1, :] *= -1
    # Return both force and full Jacobian
    return ca.Function("Jac", [theta1, theta2], [J])

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


if __name__ == "__main__":
    compute_dynamics_components = get_dynamics_components()
    casadi_force_fn_reg = build_force_function_reg()
    casadi_force_fn_unreg = build_force_function_unreg()
    casadi_force_fn_unreg_simpl = build_force_function_unregularized()
    casadi_end_effector_dynamic_components = get_ee_dynamic_components()
    compute_jacobian_casadi = build_jacobian_casadi_function()
    force_from_state_and_torque_fn = complete_force()

    states = load_data(TRAJECTORIES, NUM_TRAJECTORIES, TRAJECTORY_LEN + 1)
    actions = load_data(ACTIONS, NUM_TRAJECTORIES, TRAJECTORY_LEN)

with open("get_ee_dynamic_componentes_dataset_final.csv", "w", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["traj_idx", "step", "force_norm", "torque_norm", "jacobian_condition"])

    for traj_idx in range(states.shape[0]):
        print(f"The initial states are: {states[traj_idx, 0]}")
        for i, joint in enumerate(controllable_joints):
            p.resetJointState(robot_id, joint, states[traj_idx, 0, i], states[traj_idx, 0, 2+i])
        for step in range(actions.shape[1]):
            try:
                state = states[traj_idx, step+1]
                torque = actions[traj_idx, step]    
                theta1, theta2 = state[0], state[1]
                dtheta1, dtheta2 = state[2], state[3]
                F_e, J, M, C, G, F_damping = casadi_end_effector_dynamic_components(theta1, theta2, dtheta1, dtheta2, torque)


                # print(f"F_e: {F_e}")

                # F_e_2 = force_from_state_and_torque_fn(state, torque)
                # print(f"F_e_2: {F_e_2}")


                
                # print(f"J: {J}") 
                # print(f"M: {M}")
                # print(f"C: {C}")
                # print(f"G: {G}")
                # print(f"F_damping: {F_damping}")

                # get_jacobian_pybullet([theta1, theta2], [dtheta1, dtheta2],torque)

                # # CasADi dynamics
                # M_cas, C_cas, G_cas, D_cas = compute_dynamics_components(theta1, theta2, dtheta1, dtheta2)
                # J_cas = compute_jacobian_casadi(theta1, theta2)

                # # Convert to NumPy
                # C_cas = C_cas.full().flatten()
                # G_cas = G_cas.full().flatten()
                # D_cas = D_cas.full().flatten()
                # J_cas = J_cas.full()
                cond_J = np.linalg.cond(J)

                # # Force estimation
                # F_reg = casadi_force_fn_reg(J_cas, torque, C_cas, G_cas, D_cas)
                # F_simple =  np.linalg.norm(casadi_force_fn_unreg_simpl(J_cas, torque))
                # F_reg_numpy = F_reg.full().flatten()
                # F_reg_e = np.linalg.norm(F_reg_numpy)

                tau_norm = np.linalg.norm(torque)

                F_test = np.linalg.norm(force_from_state_and_torque_fn(state, torque))

                # Write to file
                writer.writerow([traj_idx, step, F_test, tau_norm, cond_J])

            except Exception as e:
                print(f"[ERROR] traj {traj_idx} | step {step}: {str(e)}")


            """
            # CasADi dynamics
            M_cas, C_cas, G_cas, D_cas = compute_dynamics_components(theta1, theta2, dtheta1, dtheta2)
            J_cas = compute_jacobian_casadi(theta1, theta2)

            # Convert CasADi to NumPy
            M_cas = M_cas.full()
            C_cas = C_cas.full().flatten()
            G_cas = G_cas.full().flatten()
            D_cas = D_cas.full().flatten()
            J_cas = J_cas.full()

            # Check Jacobian condition number
            cond_J = np.linalg.cond(J_cas)

            # Force estimation
            F = casadi_force_fn_with_dynamics(J_cas, torque, C_cas, G_cas, D_cas)
            F_reg = casadi_force_fn(J_cas, torque, C_cas, G_cas, D_cas)

            F_numpy = F.full().flatten()
            F_reg_numpy = F_reg.full().flatten()
            
            F_e = np.linalg.norm(F_numpy)
            F_reg_e = np.linalg.norm(F_reg_numpy)

            print(f"[traj {traj_idx} | step {step}] F_e: {F_e:.4f} | F_reg_e: {F_reg_e:.4f}")
            if F_e > 35:
                if cond_J > THRESH_COND:
                    print(f"[traj {traj_idx} | step {step}] ⚠️ High Jacobian condition number: {cond_J:.2f}") 
                print(f"The control actions were: {torque}")

            print(f"[traj {traj_idx} | step {step}] ✅ F_e: {F_e:.4f}")
            """

