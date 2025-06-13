import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

NUM_TRAJ = 10000

# === CONFIG ===
print(__file__)
csv_path = str(Path(__file__).resolve().parents[1]) + "/mpc_generated_dataset.csv"
output_folder = "analysis_outputs/" + str(NUM_TRAJ)
os.makedirs(output_folder, exist_ok=True)

trajectories = []

with open(csv_path, 'r') as f:
    for line in f:
        line = line.strip()
        steps = line.split(',')
        traj = []
        for step in steps:
            if len(step.split(';')) != 4:
                continue
            values = [float(x) for x in step.split(';')]
            traj.append(values)
        trajectories.append(traj)

data = np.array(trajectories)[:NUM_TRAJ]  # (n_traj, 500, 4)

print(f"Loaded {data.shape[0]} trajectories, each with {data.shape[1]} steps and {data.shape[2]} variables")

pos1 = data[:, :, 0].flatten()
pos2 = data[:, :, 1].flatten()
vel1 = data[:, :, 2].flatten()
vel2 = data[:, :, 3].flatten()

# === 1. STATISTICS ===
stats = pd.DataFrame({
    "Variable": ['pos1', 'pos2', 'vel1', 'vel2'],
    "Mean": [pos1.mean(), pos2.mean(), vel1.mean(), vel2.mean()],
    "Std": [pos1.std(), pos2.std(), vel1.std(), vel2.std()],
    "Min": [pos1.min(), pos2.min(), vel1.min(), vel2.min()],
    "Max": [pos1.max(), pos2.max(), vel1.max(), vel2.max()],
})
stats.to_csv(f"{output_folder}/statistics.csv", index=False)
print(stats)

variables = {'pos1': pos1, 'pos2': pos2, 'vel1': vel1, 'vel2': vel2}
names = {'pos1': r'$\theta_{1}$ [rad]', 'pos2':  r'$\theta_{2}$ [rad]', 'vel1': r'$\dot\theta_{1}$ [rad/s]', 'vel2': r'$\dot\theta_{2}$ [rad/s]'}
title_names = {'pos1': r'$\theta_{1}$', 'pos2':  r'$\theta_{2}$', 'vel1': r'$\dot\theta_{1}$', 'vel2': r'$\dot\theta_{2}$'}
# === 2. HISTOGRAMS ===
for name, values in variables.items():
    plt.figure(figsize=(8, 6))
    sns.histplot(values, bins=100, kde=True)
    plt.title(f'Histogram of {names[name]}')
    plt.xlabel(names[name])
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/hist_{name}.pdf", format="pdf", bbox_inches='tight')
    plt.close()


# === 3. 2D HEXBIN PLOTS (Only pos1 vs pos2 and vel1 vs vel2) ===
pairs = [('pos1', 'pos2'), ('vel1', 'vel2')]

for x_name, y_name in pairs:
    plt.figure(figsize=(8, 6))

    if (x_name, y_name) == ('vel1', 'vel2'):
        x_limits = np.percentile(variables[x_name], [1, 99])
        y_limits = np.percentile(variables[y_name], [1, 99])
        plt.xlim(x_limits)
        plt.ylim(y_limits)

    plt.hexbin(variables[x_name], variables[y_name], gridsize=100, cmap='inferno')
    plt.colorbar(label='Counts')
    plt.xlabel(names[x_name])
    plt.ylabel(names[y_name])
    plt.title(f'{title_names[x_name]} vs {title_names[y_name]}')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/heatmap_{x_name}_{y_name}.pdf", format="pdf", bbox_inches='tight')
    plt.close()


# === 4. EE Position Calculation ===
# Link lengths
l1, l2 = 1.0, 1.0

# Convert pos1, pos2 to EE x, y
# Joint 1 is from vertical (up), Joint 2 is relative and inverted
theta1 = np.pi - pos1 # Rotate to standard math axis
theta2 = -pos2           # Inverted relative angle

x_ee = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
y_ee = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)

# === 5. EE Heatmap ===
plt.figure(figsize=(8, 6))
plt.hexbin(x_ee, y_ee, gridsize=100, cmap='inferno')
plt.colorbar(label='Counts')
plt.xlabel('X position [m]')
plt.ylabel('Y position [m]')
plt.ylim(bottom=0)
plt.title('End-Effector Position Heatmap')
plt.tight_layout()
plt.savefig(f"{output_folder}/heatmap_ee_position.pdf", format="pdf", bbox_inches='tight')
plt.close()

print(f"All outputs saved to: {output_folder}")
