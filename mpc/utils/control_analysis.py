import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# ——— Configuration ————————————————————————————————————————————
# Here, a loop could be added to iterate over different weights and save the results accordingly.
n = 3
weight = 0.01

BASE_DIR = "/home/jeauscq/Desktop/jcq_thesis/"
EXP_DIR = BASE_DIR+ f"mpc/weight_studies/track_vs_ctrleff/weight_study_{n}/experiment_w{weight}/"
input_file = EXP_DIR + "mpc_generated_actions.csv"
histogram_output = EXP_DIR + "joint_torque_histograms_rescaled.pdf"
stats_output = EXP_DIR + "torque_statistics.txt"

# ——— Load and parse the control dataset —————————————————————————
trajectories = []
with open(input_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        trajectory = [list(map(float, t.split(";"))) for t in row]
        trajectories.append(trajectory)

data = np.array(trajectories)  # shape: (num_trajectories, time_steps, 2)
joint1_torques = data[:, :, 0].flatten()
joint2_torques = data[:, :, 1].flatten()

# ——— Plot histograms ————————————————————————————————————————
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(joint1_torques, bins=150, color="blue", alpha=0.75)
plt.xlim(-40, 40)
plt.title("Joint 1 Torque Histogram")
plt.xlabel("Torque")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(joint2_torques, bins=150, color="green", alpha=0.75)
plt.xlim(-20, 20)
plt.title("Joint 2 Torque Histogram")
plt.xlabel("Torque")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(histogram_output)
plt.close()

# ——— Compute statistics —————————————————————————————————————
def compute_stats(arr):
    return {
        "min": np.min(arr),
        "max": np.max(arr),
        "mean": np.mean(arr),
        "std": np.std(arr)
    }

joint1_stats = compute_stats(joint1_torques)
joint2_stats = compute_stats(joint2_torques)

# ——— Write stats to a text file ———————————————————————————————
with open(stats_output, "w") as f:
    f.write("Torque Statistics for Joint 1:\n")
    for k, v in joint1_stats.items():
        f.write(f"{k}: {v:.4f}\n")

    f.write("\nTorque Statistics for Joint 2:\n")
    for k, v in joint2_stats.items():
        f.write(f"{k}: {v:.4f}\n")

print("Histogram saved to:", histogram_output)
print("Statistics saved to:", stats_output)
