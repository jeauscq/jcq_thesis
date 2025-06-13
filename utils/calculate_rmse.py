import pandas as pd
import numpy as np

# Load your CSV file
file_path = '/home/jeauscq/Desktop/jcq_thesis/experiments_tuning/testtttttttt/_detailed_trajectories.csv'  # Replace with the actual path
df = pd.read_csv(file_path)

# Compute squared errors
squared_errors = {
    'pos_1': (df['actual_pos_1'] - df['desired_pos_1']) ** 2,
    'pos_2': (df['actual_pos_2'] - df['desired_pos_2']) ** 2,
    'vel_1': (df['actual_vel_1'] - df['desired_vel_1']) ** 2,
    'vel_2': (df['actual_vel_2'] - df['desired_vel_2']) ** 2,
}

# Compute RMSE for each quantity
rmse = {key: np.sqrt(np.mean(err)) for key, err in squared_errors.items()}

# Print results
print(f"RMSE Position Joint 1: {rmse['pos_1']:.4f}")
print(f"RMSE Position Joint 2: {rmse['pos_2']:.4f}")
print(f"RMSE Velocity Joint 1: {rmse['vel_1']:.4f}")
print(f"RMSE Velocity Joint 2: {rmse['vel_2']:.4f}")
