# This script generates plots for each trajectory in the dataset, showing the relationship between various state variables and the force norm.
# It also includes a line plot of the force norm over time.


import pandas as pd
import matplotlib.pyplot as plt
import os

# Load datasets
df = pd.read_csv("/home/jeauscq/Desktop/jcq_thesis/force_dataset_final.csv", delimiter=",")
df2 = pd.read_csv("/home/jeauscq/Desktop/jcq_thesis/datasets/Policy/MPC/torque_Const/mpc_generated_tor_constraint_dataset_acc.csv")

# Output directory (optional)
output_dir = "trajectory_plots"
os.makedirs(output_dir, exist_ok=True)
# Define state columns
state_columns = ["pos1", "pos2", "vel1", "vel2", "acc1", "acc2"]

# Iterate through all trajectories
for traj_idx in range(150):
    try:
        # Extract structured state data
        trajectory_data = []
        for i in range(1, len(df2.columns)):
            state_str = df2.iloc[traj_idx, i]
            state = [float(x) for x in state_str.split(';')]
            trajectory_data.append(state)

        trajectory_df = pd.DataFrame(trajectory_data, columns=state_columns)

        # Get force data for this trajectory
        df_traj = df[df["traj_idx"] == traj_idx].reset_index(drop=True)
        if len(df_traj) != len(trajectory_df):
            continue  # skip if length mismatch

        # Add jacobian_condition
        trajectory_df["jacobian_condition"] = df_traj["jacobian_condition"]
        trajectory_df["torque_norm"] = df_traj["torque_norm"]
        trajectory_df["force_norm"] = df_traj["force_norm"]
        
        # Prepare plots
        fig, axs = plt.subplots(3, 3, figsize=(18, 12))
        axs = axs.flatten()

        for i, col in enumerate(state_columns + ["jacobian_condition"] + ["torque_norm"]):
            axs[i].scatter(trajectory_df[col], df_traj["force_norm"], s=10)
            axs[i].set_xlabel(col)
            axs[i].set_ylabel("Force Norm")
            axs[i].set_title(f"Force Norm vs {col}")
            axs[i].grid(True)

       # 9: Line plot of torque_norm over time
        axs[8].plot(trajectory_df["force_norm"])
        axs[8].set_title("Force Norm over Time")
        axs[8].set_xlabel("Time Step")
        axs[8].set_ylabel("Force Norm")
        axs[8].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"trajectory_{traj_idx}.png"))
        plt.close()

    except Exception as e:
        print(f"Skipping trajectory {traj_idx} due to error: {e}")