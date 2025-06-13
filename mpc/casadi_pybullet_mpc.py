# This script implements a Model Predictive Control (MPC) scheme using CasADi.A symbolic model of the 2-DoF robot is defined via the `define_dynamics` function.
# For improved accuracy, physical parameters are loaded from the URDF file, including: # link lengths, inertias, center of mass positions, masses, and joint damping coefficients.

# The MPC controller receives the current state and the next H desired states as input. # It then optimizes a cost function over a horizon of H steps, producing a sequence of
# 10 optimal control actions to minimize the trajectory tracking error.

# Only the first control action is applied, and the optimization is repeated at the next step, following the receding horizon principle.

# The outputs are stored in three files:
#    - One for the executed trajectories (joint positions and velocities),
#    - One for the control inputs (torques),
#    - One listing failed trajectories, if any.

# In this implementation, the cost function penalizes both trajectory tracking errors and control effort. Control effort is weighted using a vector since the control actions have an 
# evidently different scale, therefore, they are penalized accordingly. The value of the weights will require a study to determine.

# Note that this is an improved setup: the symbolic model is used only for prediction, while PyBullet is used as the one used to simulate the robot, meaning there IS a slight model mismatch. As a result,
# trajectory tracking is more realistic.

# The formulas used to symbolically model the 2DoF robot come from [A. A. Okubanjo, et al., “Modeling of 2 dof robot arm and control,” Futo Journal Series, vol. 3, no. 2, 2017]

import casadi as ca
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
import numpy as np
import time
import csv
import sys
import itertools
from pathlib import Path
from urdfpy import URDF


# ——— Project Modules ————————————————————————————————————————
current_file = Path().resolve()
while current_file.name != "jcq_thesis" and current_file != current_file.parent:
    current_file = current_file.parent
sys.path.append(str(current_file))

from utils.config import BASE_DIR

# BASE_DIR = "/home/jeauscq/Desktop/jcq_thesis/"
DATASETS_DIR = BASE_DIR + "datasets/"
URDF_FILE = BASE_DIR + '/models/2dof_planar_robot.urdf'
TRAJECTORIES = DATASETS_DIR + "Testing/test_trajectories.csv"

# OUTPUT_TRAJECTORIES= DATASETS_DIR + "mpc_generated_unconstrained_expert_dataset.csv" # ESTAS SON LAS QUE ESTOY USANDO PARA POS_VEL DE MPC DISCRIMINATOR
# OUTPUT_TRAJECTORIES= DATASETS_DIR + "Discriminator/MPC/torque_Const/mpc_generated_tor_constraint_expert_dataset.csv"
OUTPUT_TRAJECTORIES = DATASETS_DIR + "Testing/MPC/unconst/w0.01/test_trajectories.csv"
# OUTPUT_TRAJECTORIES= DATASETS_DIR + "mpc_generated_constrained_dataset.csv"


# OUTPUT_CONTROL_ACTIONS = DATASETS_DIR + "mpc_generated_unconstrained_expert_actions.csv" # ESTAS SON LAS QUE ESTOY USANDO PARA POS_VEL DE MPC DISCRIMINATOR
# OUTPUT_CONTROL_ACTIONS = DATASETS_DIR + "Discriminator/MPC/torque_Const/mpc_generated_tor_constrained_expert_actions.csv"
OUTPUT_CONTROL_ACTIONS = DATASETS_DIR + "Testing/MPC/unconst/w0.01/test_actions.csv"
# OUTPUT_CONTROL_ACTIONS = DATASETS_DIR + "mpc_generated_constrained_actions.csv"

# OUTPUT_FAILED_TRAJECTORIES = DATASETS_DIR + "failed_unconstrained_expert_trajectories.txt" # ESTAS SON LAS QUE ESTOY USANDO PARA POS_VEL DE MPC DISCRIMINATOR
# OUTPUT_FAILED_TRAJECTORIES = DATASETS_DIR + "Discriminator/MPC/torque_Const/failed_tor_constrained_expert_trajectories.txt"
OUTPUT_FAILED_TRAJECTORIES = DATASETS_DIR + "Testing/MPC/unconst/w0.01/failed.txt"
# OUTPUT_FAILED_TRAJECTORIES = DATASETS_DIR + "failed_constrained_trajectories.txt"


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
H = 10  # Horizon length
q1_min, q1_max = 0, 3.14  # Position limits
q2_min, q2_max = -3.14, 3.14
v1_min, v1_max = -2.8, 2.8  # Velocity limits
v2_min, v2_max = -4.0, 4.0
tau1_min, tau1_max = -20, 18  # Torque limits
tau2_min, tau2_max = -12, 12
g = 9.81  # Gravity acceleration

# Start PyBullet
client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -g)
p.setTimeStep(DT)

# Load URDF in pybullet
robot_id = p.loadURDF(URDF_FILE, basePosition=(0, 0, 0.07), #  0.07 space from the floor to prevent collisions.
                      baseOrientation=p.getQuaternionFromEuler((0, 3.14, 0)), # Rotated 180° to point upwards.
                      useFixedBase=True) # Fixed on the base

# Setup controllable joints
controllable_joints = []
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        controllable_joints.append(j)

# Disable velocity control and enables torque control
p.setJointMotorControlArray(robot_id, controllable_joints, controlMode=p.VELOCITY_CONTROL, forces=[0, 0])
p.setJointMotorControlArray(robot_id, controllable_joints, controlMode=p.TORQUE_CONTROL, forces=[0, 0])


def read_trajectories_csv(file_path):
    """Reads the CSV file row by row, reconstructing full trajectories."""
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # Split each state, convert values to float, and reconstruct trajectory
            trajectory = [list(map(float, state.split(";"))) for state in row]
            yield np.array(trajectory)  # Yield the full trajectory as a list of states


def plot_trajectories(reference_trajectory, actual_trajectory):
    """
    Plots joint positions and velocities for a 2-DOF robot arm.

    :param reference_trajectory: Numpy array of reference states (shape: [N, 4])
    :param actual_trajectory: Numpy array of actual states (shape: [N, 4])
    """
    time_steps = np.arange(reference_trajectory.shape[0])  # Time indices

    # Extract joint positions and velocities
    ref_theta1, ref_theta2 = reference_trajectory[:, 0], reference_trajectory[:, 1]
    ref_theta1_dot, ref_theta2_dot = reference_trajectory[:, 2], reference_trajectory[:, 3]

    act_theta1, act_theta2 = actual_trajectory[:, 0], actual_trajectory[:, 1]
    act_theta1_dot, act_theta2_dot = actual_trajectory[:, 2], actual_trajectory[:, 3]

    plt.figure(figsize=(12, 8))

    # Plot Joint 1 Position
    plt.subplot(2, 2, 1)
    plt.plot(time_steps, ref_theta1, 'r--', label="Ref θ1")
    plt.plot(time_steps, act_theta1, 'b-', label="Actual θ1")
    plt.xlabel("Time Step")
    plt.ylabel("Joint 1 Position (rad)")
    plt.ylim(-3.14, 3.14)
    plt.legend()
    plt.title("Joint 1 Position")

    # Plot Joint 2 Position
    plt.subplot(2, 2, 2)
    plt.plot(time_steps, ref_theta2, 'r--', label="Ref θ2")
    plt.plot(time_steps, act_theta2, 'b-', label="Actual θ2")
    plt.xlabel("Time Step")
    plt.ylabel("Joint 2 Position (rad)")
    plt.ylim(-3.14, 3.14)
    plt.legend()
    plt.title("Joint 2 Position")

    # Plot Joint 1 Velocity
    plt.subplot(2, 2, 3)
    plt.plot(time_steps, ref_theta1_dot, 'r--', label="Ref θ1_dot")
    plt.plot(time_steps, act_theta1_dot, 'b-', label="Actual θ1_dot")
    plt.xlabel("Time Step")
    plt.ylabel("Joint 1 Velocity (rad/s)")
    plt.ylim(-3.14, 3.14)
    plt.legend()
    plt.title("Joint 1 Velocity")

    # Plot Joint 2 Velocity
    plt.subplot(2, 2, 4)
    plt.plot(time_steps, ref_theta2_dot, 'r--', label="Ref θ2_dot")
    plt.plot(time_steps, act_theta2_dot, 'b-', label="Actual θ2_dot")
    plt.xlabel("Time Step")
    plt.ylabel("Joint 2 Velocity (rad/s)")
    plt.ylim(-3.14, 3.14)
    plt.legend()
    plt.title("Joint 2 Velocity")

    plt.tight_layout()
    plt.show()


def define_dynamics(state, control):
    """
    Defines the system dynamics for the 2-DOF planar robot.
    :param state: Current state [theta1, theta2, theta1_dot, theta2_dot]
    :param control: Control input [F1, F2]
    :return: State derivatives [theta1_dot, theta2_dot, theta1_ddot, theta2_ddot]
    """
    theta1, theta2, theta1_dot, theta2_dot = state[0], state[1], state[2], state[3]
    
    # External forces
    F1, F2 = control[0], control[1]
    F = ca.vertcat(F1, F2)

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
    F_total = F + F_damping
    
    # Compute acceleration (q_ddot)
    q_ddot = ca.solve(M, F_total - C - G)

    return ca.vertcat(theta1_dot, theta2_dot, q_ddot)


def force(state, torque):

    theta1, theta2, theta1_dot, theta2_dot = state[0], state[1], state[2], state[3],
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
    tau_ext = torque - tau_dyn

    # Use direct inverse of J^T
    F_e = ca.pinv(J.T) @ tau_ext

    return F_e


def simulate_next_state(state, control, dt=DT):
    """
    Computes the next state using numerical integration.
    :param state: Current state [theta1, theta2, theta1_dot, theta2_dot]
    :param control: Control input [F1, F2]
    :param dt: Time step for integration
    :return: Next state after dt seconds
    """
    state_var = ca.SX.sym('state', 4)
    control_var = ca.SX.sym('control', 2)
    
    f = ca.Function('f', [state_var, control_var], [define_dynamics(state_var, control_var)])
    
    # Define the integrator correctly with 'tf' inside the options dictionary
    dae = {'x': state_var, 'p': control_var, 'ode': f(state_var, control_var)}
    intg_options = {'t0': 0, 'tf': dt, 'simplify': True, 'number_of_finite_elements': 4}
    intg = ca.integrator('intg', 'rk', dae, intg_options)

    # Call the integrator normally
    res = intg(x0=state, p=control)
    
    return res['xf']


def compute_jacobian(theta1, theta2):
    """
    Computes the Jacobian of the end-effector for a 2-DoF planar arm.
    """
    J11 = -L1 * ca.sin(theta1) - L2 * ca.sin(theta1 + theta2)
    J12 = -L2 * ca.sin(theta1 + theta2)
    J21 = L1 * ca.cos(theta1) + L2 * ca.cos(theta1 + theta2)
    J22 = L2 * ca.cos(theta1 + theta2)
    
    J = ca.vertcat(
        ca.horzcat(J11, J12),
        ca.horzcat(J21, J22)
    )
    return J


def take_next_state(x_current, u_opt):
    p.resetJointState(robot_id, controllable_joints[0], x_current[0], x_current[2])
    p.resetJointState(robot_id, controllable_joints[1], x_current[1], x_current[3])
    # Apply one step of control
    p.setJointMotorControlArray(robot_id, controllable_joints, controlMode=p.TORQUE_CONTROL, forces=u_opt)
    p.stepSimulation()
    pb_pos, pb_vel = zip(*[p.getJointState(robot_id, j)[:2] for j in controllable_joints])
    return list(pb_pos) + list(pb_vel)


def initialize_ocp(N=10, dt=DT):
    """
    Initializes the optimization problem for the OCP. Here the cost function is defined.
    :param N: Horizon length
    :param dt: Time step
    :return: Initialized Opti instance and problem variables
    """
    opti = ca.Opti()
    
    x = opti.variable(4, N+1)  # State variables
    u = opti.variable(2, N)    # Control variables
    p = opti.parameter(4, 1)   # Initial state parameter
    ref = opti.parameter(4, N+1)  # Reference trajectory

    # Cost 1: Trajectory tracking
    traj_cost = ca.sumsqr(x - ref)

    # Cost 2: Control effort
    w1 = 0.01
    w0 = w1 / np.sqrt(5)
    R = ca.diag([w0, w1])  # Weights vector
    effort_cost = ca.sumsqr(R @ u) 

    # Cost function to minimize
    opti.minimize(traj_cost + effort_cost)

    # Initial condition constraint
    opti.subject_to(x[:, 0] == p)  
        
    for k in range(N):
        x_next = simulate_next_state(x[:, k], u[:, k], dt)
        opti.subject_to(x[:, k+1] == x_next)  # State update constraint
    
    # # Position and velocity constraints
    # for k in range(N+1):
    #     # Position contraints
    #     opti.subject_to(q1_min <= x[0, k])
    #     opti.subject_to(x[0, k] <= q1_max)
    #     opti.subject_to(q2_min <= x[1, k])
    #     opti.subject_to(x[1, k] <= q2_max)

    #     # Velocity contraints
    #     opti.subject_to(v1_min <= x[2, k])
    #     opti.subject_to(x[2, k] <= v1_max)
    #     opti.subject_to(v2_min <= x[3, k])
    #     opti.subject_to(x[3, k] <= v2_max)

    # Torque constraints        
    # for k in range(u.shape[1]):
        # # Enforce end-effector force magnitude constraint
        # theta1_next = x[0, k+1]
        # theta2_next = x[1, k+1]
        # J_next = compute_jacobian(theta1_next, theta2_next)
        # tau_k = u[:, k]
        
        # # Estimate the end-effector force at x_{k+1}
        # F_e = force(x, u))
        # force_norm = ca.norm_2(F_e)
        
        # # Apply conservative constraint
        # opti.subject_to(force_norm <= 60)

        # # Torque contraints
        # opti.subject_to(tau1_min <= u[0, k])
        # opti.subject_to(u[0, k] <= tau1_max)
        # opti.subject_to(tau2_min <= u[1, k])
        # opti.subject_to(u[1, k] <= tau2_max)

        # The generated dataset were generated taking into consideration:
        # 1. Positions and velocities constraints.
        # 2. Torques constraints.
        # 3. Both positions, velocities and torque constraints.


    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', opts)

    return opti, u, p, ref


def solve_ocp(opti, u, p, ref, initial_state, ref_traj):
    """
    Solves the OCP given the current state and reference trajectory.
    :param opti: Initialized Opti instance
    :param x: State variable
    :param u: Control variable
    :param p: Initial state parameter
    :param ref: Reference trajectory parameter
    :param initial_state: Current state of the system
    :param ref_traj: Desired reference trajectory
    :return: Optimized control action
    """
    start = time.time()
    opti.set_value(p, initial_state)
    opti.set_value(ref, ref_traj.T)    
    sol = opti.solve()
    end = time.time()
    return sol.value(u[:, 0]), end-start


def MPC(ref_traj, H=10, dt=DT):
    """
    Runs Model Predictive Control (MPC) using the refactored OCP solver.
    :param initial_state: Initial state [theta1, theta2, theta1_dot, theta2_dot]
    :param ref_traj: Reference trajectory (4 x N+1 matrix)
    :param H: Prediction horizon
    :param dt: Time step
    :return: Optimized control signals for the full trajectory
    """
    # Initialize storing variables
    U_log = []  # Store control signals
    times = []  # Store duration of steps
    x_current = ref_traj[0]  # Initialize state
    states_log = [ref_traj[0]] # Stores actual state

    # Initialize OCP solver
    opti, u, p, ref = initialize_ocp(H, dt)
    
    # Iterates over each step of the trajectory
    for t in range(1, len(ref_traj)):
        # Defines the sliding horizon
        if t < len(ref_traj) - H:
            x_target = ref_traj[t : t + H + 1]
        else:
            missing = (t + H + 1) - len(ref_traj) # Fills in case these are the last H steps
            x_target = np.vstack((ref_traj[t:], np.tile(ref_traj[-1], (missing, 1))))
        
        # Obtains optimized control actions and the time it took
        u_opt, step_time = solve_ocp(opti, u, p, ref, x_current, x_target)
        
        times.append(step_time)
        U_log.append(u_opt)  # Stores first u
        
        # Takes the first step
        x_current = simulate_next_state(x_current, u_opt)
        states_log.append(np.array(x_current).flatten()) # Stores actual state
    
    return np.array(U_log), states_log, np.array(times)


if __name__ == "__main__":
    # Count how many rows are already in the output file, to start where it was left the last time
    try:
        with open(OUTPUT_TRAJECTORIES, "r") as f:
            existing_rows = sum(1 for _ in f)
    except FileNotFoundError:
        existing_rows = 0  # If file doesn't exist yet
    print(f"Existing rows in output file: {existing_rows}")

    # Count how many rows failed in the previous attempt
    try:
        with open(OUTPUT_FAILED_TRAJECTORIES, "r") as f:
            failed_traj = sum(1 for _ in f)
    except FileNotFoundError:
        failed_traj = 0  # If file doesn't exist yet
    print(f"Failed trajectories: {failed_traj}")


    # Open output_file in append mode
    with open(OUTPUT_TRAJECTORIES, "a", newline="") as f:
        writer = csv.writer(f)

        # Read reference trajectories, skipping existing ones
        for idx, reference_trajectory in enumerate(itertools.islice(read_trajectories_csv(TRAJECTORIES), existing_rows+failed_traj, None)):
            traj_index = existing_rows + failed_traj + idx + 1
            if idx%100==0:
                print(f"Solving trajectory number: {traj_index}")

            # This try-except was implemented because if the simulator deviates too much from the actual model, the optimizer
            # fails and breaks the run. With this solution it only skips this trajectory. The first step of the trajectory is skipped
            # because the model is not able to predict a reasonably next state, and the optimizer fails.
            try:
                optimized_controls, registered_log, times = MPC(reference_trajectory[1:])
            except Exception as e:
                print(f"Skipping trajectory {traj_index} due to exception in MPC(). Reason: {e}")
                with open(OUTPUT_FAILED_TRAJECTORIES, "a") as logf:
                    logf.write(f"{traj_index}\n")
                continue  # Skip to next trajectory
            registered_log = np.vstack(registered_log)

            # Save taken trajectory to traj CSV
            formatted_traj = ["{:.6f};{:.6f};{:.6f};{:.6f}".format(*step) for step in registered_log]
            writer.writerow(formatted_traj)

            # Save control actions
            formatted_controls = ["{:.6f};{:.6f}".format(f1, f2) for f1, f2 in optimized_controls]
            with open(OUTPUT_CONTROL_ACTIONS, "a", newline="") as cf:
                control_writer = csv.writer(cf)
                control_writer.writerow(formatted_controls)
