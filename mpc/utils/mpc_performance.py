# To do:
# Consider and skip failed trajectories. They are stored in a separate file.

import csv
import numpy as np
import matplotlib.pyplot as plt

# File paths (adjust if needed)
ref_path = "/home/jeauscq/Desktop/jcq_thesis/datasets/Policy/training_trajectories_policy.csv"
mpc_path = "/home/jeauscq/Desktop/jcq_thesis/datasets/Policy/MPC/torque_Const/2iteration/mpc_generated_tor_2_constrained_dataset.csv"

# Load and parse a trajectory row
def parse_trajectory(row):
    return np.array([list(map(float, state.split(";"))) for state in row])

# Read both files
with open(ref_path, "r") as f_ref, open(mpc_path, "r") as f_mpc:
    ref_rows = list(csv.reader(f_ref))
    mpc_rows = list(csv.reader(f_mpc))

pos_rmses = []
vel_rmses = []

# Iterate through corresponding trajectories. The first state was skipped in the
# trajectories because the mpc was not capable of considering the required control 
# action to reasonably set the robot in motion.
for ref_row, mpc_row in zip(ref_rows, mpc_rows):
    ref_traj = parse_trajectory(ref_row)[1:]  # Skip first state
    mpc_traj = parse_trajectory(mpc_row)      # Already 499 steps

    if len(ref_traj) == len(mpc_traj) == 499:
        pos_rmse = np.sqrt(np.mean((ref_traj[:2] - mpc_traj[:2]) ** 2))
        vel_rmse = np.sqrt(np.mean((ref_traj[2:] - mpc_traj[2:]) ** 2))
        pos_rmses.append(pos_rmse)
        vel_rmses.append(vel_rmse)

# # Plot histogram
# plt.figure(figsize=(8, 5))
# plt.hist(rmses, bins=40, edgecolor='black')
# plt.xlabel("RMSE per trajectory")
# plt.ylabel("Frequency")
# plt.title("RMSE Distribution Across Trajectories")
# plt.tight_layout()
# plt.savefig("rmse_histogram.pdf")
# plt.show()

# Print stats
print(f"Number of trajectories: {len(pos_rmses)}")

print(f"Min pos RMSE: {np.min(pos_rmses):.6f}")
print(f"Max pos RMSE: {np.max(pos_rmses):.6f}")
print(f"Mean pos RMSE: {np.mean(pos_rmses):.6f}")
print(f"Std pos RMSE: {np.std(pos_rmses):.6f}\n")
print(f"Min vel RMSE: {np.min(vel_rmses):.6f}")
print(f"Max vel RMSE: {np.max(vel_rmses):.6f}")
print(f"Mean vel RMSE: {np.mean(vel_rmses):.6f}")
print(f"Std vel RMSE: {np.std(vel_rmses):.6f}")