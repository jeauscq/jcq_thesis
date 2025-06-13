import numpy as np
import csv

BASE_DIR = "/home/jeauscq/Desktop/jcq_thesis/datasets/"

# Input and output files
input_csv = BASE_DIR + "/Testing/MPC/unconst/w0.02/test_trajectories.csv"
output_csv = input_csv[:-4] + "_n.csv"
stats_txt = output_csv[:-4] + "_stats.txt"

# Load and parse
trajectories = []
with open(input_csv, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        # Each row is a full trajectory of T steps: "q1;q2;dq1;dq2,q1;q2;dq1;dq2,..."
        states = [list(map(float, state_str.split(";"))) for state_str in row]
        trajectories.append(states)

trajectories = np.array(trajectories)  # Shape: [num_traj, T, 4]
flattened = trajectories.reshape(-1, 4)  # [num_traj * T, 4]

# Normalize
mean = flattened.mean(axis=0)
std = flattened.std(axis=0)
normalized = (flattened - mean) / (std + 1e-8)
normalized_trajectories = normalized.reshape(trajectories.shape)

# Save normalized CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    for traj in normalized_trajectories:
        row = [f"{step[0]:.6f};{step[1]:.6f};{step[2]:.6f};{step[3]:.6f}" for step in traj]
        writer.writerow(row)

# Save stats
with open(stats_txt, "w") as f:
    f.write("Feature:       q1         q2       dq1       dq2\n")
    f.write("mean:    " + "   ".join(f"{m:.6f}" for m in mean) + "\n")
    f.write("std:     " + "   ".join(f"{s:.6f}" for s in std) + "\n")
