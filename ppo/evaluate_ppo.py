import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import contextlib
import io
import time
import csv
from itertools import islice
from datetime import datetime
from gymnasium.envs.registration import register

try:
    from ppo.envs.CustomArmDiscr import Custom2DoFEnv
    from ppo.utils.agent import Agent
except:
    print("Could not load the local packages")


DIM_SHAPE = 4


def read_trajectories_csv(trajectories_path):
    """
    Reads the CSV file containing joint trajectories.
    Each row represents a full trajectory with semicolon-separated joint states.

    Yields:
        trajectory (list of list of float): Each trajectory is a list of joint states
        where each joint state is [joint1_pos, joint2_pos, joint1_vel, joint2_vel].
    """
    with open(trajectories_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            trajectory = [list(map(float, state.split(";"))) for state in row]
            yield trajectory


def evaluate_policy(
    config: dict,
    model_path: str,
    trajectories_path: str,
    base_dir: str = "/home/detagoy/student_projects/jcq_thesis/",
    run_name: str = "MISSING_RUN_NAME",
    step: str = "",
    final: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_traj = 200,
    policy = None,
    normalize_stats_path = "MISSING_NORMALIZE_STATS_PATH",
    time_experiment = False,
    detailed_report = False,
    clip=False,
    positions=None,
    fps=60,
):
  
    trajectories = list(islice(read_trajectories_csv(trajectories_path), num_traj))

    if time_experiment:
        parent_folder = "time_experiment"
    elif final:
        parent_folder = "experiments_tuning"
    else:
        parent_folder = "experiments_tuning_checkpoints"
    
    save_dir = base_dir + f"/{parent_folder}/{run_name}/"
    # Output path
    csv_path = os.path.join(save_dir, "all_evaluation_summary.csv")


    os.makedirs(save_dir, exist_ok=True)

    if policy == None:

        # Load policy from memory
        agent = Agent(hidden_dim=config["hidden_dim"], std_init=list([config["std_init"], config["std_init"]/3])).to(device)
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        agent.eval()
        # Reports hyperparameter and architecture
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            agent.print_actor_architecture()
        with open(os.path.join(save_dir, "description.txt"), 'w') as f:
            f.write("Policy Architecture Description:\n\n")
            f.write(buffer.getvalue() + "\n\n")

            f.write(f"Total trainable parameters:\n")
            f.write(f"- Policy: {config.get('num_params_policy', 'N/A')}\n")
            f.write(f"- Discriminator: {config.get('num_params_disc', 'N/A')}\n\n")

            f.write("Experiment Configuration:\n")
            f.write(f"- Environment: {config.get('env_id', 'N/A')}\n")
            f.write(f"- Model saved at: {model_path}\n\n")

            f.write("Hyperparameters:\n")
            for key in sorted(config.keys()):
                value = config[key]
                if key == "std_init":
                    value = f"{value} -> [{value}, {value/3}]"
                f.write(f"- {key}: {value}\n")

            f.write("\nNotes:\n")
            f.write("- `std_init` is expanded internally as a pair [std, std/2]\n")
            f.write("- `N` is the number of past states, `M` is the number of future desired states\n")
    else:
        agent = policy
        
    if detailed_report:
        trajs_file = open(os.path.join(save_dir, f"{step}_detailed_trajectories.csv"), "w", newline="")
        actions_file = open(os.path.join(save_dir, f"{step}_detailed_actions.csv"), "w", newline="")
        trajs_writer = csv.writer(trajs_file)
        actions_writer = csv.writer(actions_file)
        # Write headers
        trajs_writer.writerow([
            "trajectory_idx", "timestep", 
            "actual_pos_1", "actual_pos_2", "actual_vel_1", "actual_vel_2",
            "desired_pos_1", "desired_pos_2", "desired_vel_1", "desired_vel_2"
        ])
        actions_writer.writerow([
            "trajectory_idx", "timestep", "action_1", "action_2"
        ])

    summary_stats = []
    all_pos_errors = []
    all_vel_errors = []
    all_time_steps = []

    for idx, traj in enumerate(trajectories):
        env = gym.make("Custom2DoF-v0", trajectories=[traj], M=config["M"], N=config["N"], normalize_path=normalize_stats_path, rewardMode=2, positions=positions,
                    idx=idx, render_mode="human", fps=fps) 
                    # render_mode="human", fps=fps)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space        env = gym.wrappers.RecordEpisodeStatistics(env)....
        if clip:
            env=gym.wrappers.ClipAction(env)

        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).view(1, config["M"] + config["N"] + 1, DIM_SHAPE).to(device)

        actual_positions, desired_positions, actual_velocities, desired_velocities, actions_taken = [], [], [], [], []
        times = []
        reason = "completed"

        timestep = 0
        while True:
            if time_experiment:
                # Measure the time taken for the action
                torch.cuda.synchronize()
                start_time = time.time()
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs)
                torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
            else:
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs)
            obs_np, _, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            obs_np = obs_np.reshape(-1, 4)  # reshape (72,) -> (18, 4)

            actual_positions.append(env.unwrapped.current_state[:2])
            actual_velocities.append(env.unwrapped.current_state[2:])

            desired_positions.append(env.unwrapped.trajectory[env.unwrapped.traj_step][:2])
            desired_velocities.append(env.unwrapped.trajectory[env.unwrapped.traj_step][2:])
            if clip:
                action = np.clip(action.cpu().numpy()[0], env.action_space.low, env.action_space.high)
                actions_taken.append(action)
            else:
                actions_taken.append(action.cpu().numpy()[0])

            if detailed_report:
                # Write trajectory and action for this step
                trajs_writer.writerow([
                    idx, timestep,
                    env.unwrapped.current_state[0], env.unwrapped.current_state[1],
                    env.unwrapped.current_state[2], env.unwrapped.current_state[3],
                    env.unwrapped.trajectory[env.unwrapped.traj_step][0], env.unwrapped.trajectory[env.unwrapped.traj_step][1],
                    env.unwrapped.trajectory[env.unwrapped.traj_step][2], env.unwrapped.trajectory[env.unwrapped.traj_step][3],])
                if clip:
                    actions_writer.writerow([idx, timestep, action[0], action[1]])
                else:
                    actions_writer.writerow([idx, timestep, action.cpu().numpy()[0][0], action.cpu().numpy()[0][1]])
                timestep += 1

            if "early_termination_reason" in info:
                reason = info["early_termination_reason"]

            obs = torch.tensor(obs_np, dtype=torch.float32).view(1, config["M"] + config["N"] + 1, DIM_SHAPE).to(device)
            if terminated or truncated:
                break

        pos_errors = np.linalg.norm(np.array(actual_positions) - np.array(desired_positions), axis=1)
        vel_errors = np.linalg.norm(np.array(actual_velocities) - np.array(desired_velocities), axis=1)

        all_pos_errors.extend(pos_errors)
        all_vel_errors.extend(vel_errors)

        if time_experiment:
            all_time_steps.extend(times)
            time_avg = np.mean(times)
        
        rmse_pos = np.sqrt(np.mean(np.square(all_pos_errors)))
        rmse_vel = np.sqrt(np.mean(np.square(all_vel_errors)))

        
        if time_experiment:
            summary_stats.append({"trajectory_idx": idx, "steps": len(actual_positions), "pos_rmse": rmse_pos, "vel_rmse": rmse_vel, "time_per_step": time_avg, 
                            "terminated_by_collision": "collision" in reason, "termination_reason": reason})
        else:
            summary_stats.append({"trajectory_idx": idx, "steps": len(actual_positions), "pos_rmse": rmse_pos, "vel_rmse": rmse_vel,
                            "terminated_by_collision": "collision" in reason, "termination_reason": reason})
    
        if idx % 10 == 0 and policy == None:
            actual_positions = np.array(actual_positions)
            desired_positions = np.array(desired_positions)
            actual_velocities = np.array(actual_velocities)
            desired_velocities = np.array(desired_velocities)
            actions_taken = np.array(actions_taken)
            taken_time = np.arange(len(actual_positions))

            fig, axs = plt.subplots(3, 2, figsize=(12, 9))

            axs[0, 0].plot(taken_time, desired_positions[:, 0], 'r--', label="Desired θ1")
            axs[0, 0].plot(taken_time, actual_positions[:, 0], 'b-', label="Actual θ1")
            axs[0, 0].set_title("Joint 1 Position [rad]")
            axs[0, 0].legend()

            axs[0, 1].plot(taken_time, desired_positions[:, 1], 'r--', label="Desired θ2")
            axs[0, 1].plot(taken_time, actual_positions[:, 1], 'b-', label="Actual θ2")
            axs[0, 1].set_title("Joint 2 Position [rad]")
            axs[0, 1].legend()

            axs[1, 0].plot(taken_time, desired_velocities[:, 0], 'r--', label="Desired θ1_dot")
            axs[1, 0].plot(taken_time, actual_velocities[:, 0], 'b-', label="Actual θ1_dot")
            axs[1, 0].set_title("Joint 1 Velocity [rad/s]")
            axs[1, 0].legend()

            axs[1, 1].plot(taken_time, desired_velocities[:, 1], 'r--', label="Desired θ2_dot")
            axs[1, 1].plot(taken_time, actual_velocities[:, 1], 'b-', label="Actual θ2_dot")
            axs[1, 1].set_title("Joint 2 Velocity [rad/s]")
            axs[1, 1].legend()

            axs[2, 0].plot(taken_time, actions_taken[:, 0], label="Torque 1")
            axs[2, 1].plot(taken_time, actions_taken[:, 1], label="Torque 2")
            axs[2, 0].set_title("Joint 1 Torque [Nm]")
            axs[2, 1].set_title("Joint 2 Torque [Nm]")
            axs[2, 0].legend()
            axs[2, 1].legend()

            for ax in axs.flat:
                ax.set_xlabel("Timestep")

            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, f"trajectory_{idx:04d}_tracking.pdf"))
            plt.close(fig)

        env.close()

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(save_dir, f"{step}_policy_statistics.txt"), sep="\t", index=False)

    success_rate = np.mean(summary_df["termination_reason"] == "completed") * 100
    failure_rate = np.mean(summary_df["terminated_by_collision"]) * 100

    # Data to save
    row_data = {
        "evaluation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "step": step,
        "num_trajectories": len(summary_df),
        "success_rate_percent": round(success_rate, 2),
        "collision_rate_percent": round(failure_rate, 2),
        "rmse_position": round(rmse_pos, 4),
        "position_std": round(np.std(all_pos_errors), 4),
        "position_max": round(np.max(all_pos_errors), 4),
        "rmse_velocity": round(rmse_vel, 4),
        "velocity_std": round(np.std(all_vel_errors), 4),
        "velocity_max": round(np.max(all_vel_errors), 4),
    }

    if time_experiment:
        row_data.update({"avg_time_per_step": round(time_avg, 4),
                        "avg_time_per_step_std" : round(np.std(all_time_steps), 4),
                        "avg_time_per_step_max" : round(np.max(all_time_steps), 4)})

    # Append to CSV, writing header only if file does not exist
    if os.path.exists(csv_path):
        new_df = pd.DataFrame([row_data])
        new_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        pd.DataFrame([row_data]).to_csv(csv_path, index=False)
    
    return rmse_pos, rmse_vel, success_rate

if __name__ == "__main__":
    from pathlib import Path
    import sys
    current_file = Path().resolve()
    while current_file.name != "jcq_thesis" and current_file != current_file.parent:
        current_file = current_file.parent
    sys.path.append(str(current_file))

    from ppo.envs.CustomArmDiscr import Custom2DoFEnv
    from ppo.utils.agent import Agent

    register(id="Custom2DoF-v0", entry_point="ppo.envs.CustomArmDiscr:Custom2DoFEnv",)  # "module_path:ClassName"

    config = {'anneal_lr': False, 'ent_coef': 0.0001815466939005705, 'batch_size': 2048, 'num_minibatches': 64,
            'num_envs': 16, 'update_epochs': 7, 'hidden_dim': 256, 'num_heads': 8, 'lambda_gp': 8, 'batch_disc': 4096,
            'minibatch_disc': 1024, 'std_init': 3.0, 'lr_pol': 5e-05, 'lr_val': 0.00044283380419313267, 'lr_disc': 4.1158243033544476e-05,
            'gamma': 0.975, 'N': 5, 'M': 11, 'gae_lambda': 0.95, 'clip_coef': 0.2, 'clip_vloss': True, 'vf_coef': 0.5, 'max_grad_norm': 0.5,
            'target_kl': None, 'norm_adv': True, 'env_id': 'Custom2DoF-v0', 'num_params_policy': 283717, 'num_params_disc': 220008}

    config["num_envs"] = int(config["num_envs"])
    config["num_minibatches"] = int(config["num_minibatches"]) 
    config["batch_size"] = int(config["batch_size"])
    config["update_epochs"] = int(config["update_epochs"])
    config["num_heads"] = int(config["num_heads"])
    config["batch_disc"] = int(config["batch_disc"])
    config["minibatch_disc"] = int(config["minibatch_disc"])
    config["N"] = int(config["N"])
    config["M"] = int(config["M"])
    config["anneal_lr"] = bool(config["anneal_lr"])
    config["clip_vloss"] = bool(config["clip_vloss"])
    config["norm_adv"] = bool(config["norm_adv"])
    config["clip_coef"] = float(config["clip_coef"])
    config["gae_lambda"] = float(config["gae_lambda"])
    config["vf_coef"] = float(config["vf_coef"])
    config["max_grad_norm"] = float(config["max_grad_norm"])
    
    for j in ["_int"]:
        for i in ['cuda']:
            evaluate_policy(
                config=config,
                model_path= f"/home/jeauscq/Desktop/ResultsThesis/3.Const/rewardMode2/Softer/experiment043/runs/trial_20250529_102409/models_and_optimizers{j}.pt",
                trajectories_path="/home/jeauscq/Desktop/jcq_thesis/datasets/Testing/test_trajectories.csv",
                base_dir= "/home/jeauscq/Desktop/expss",  
                run_name= f"report{j}_{i}",
                step= "",
                final= False,
                device= i,
                num_traj= 1100,
                normalize_stats_path= "/home/jeauscq/Desktop/jcq_thesis/datasets/Testing/test_trajectories_n_stats.txt",
                time_experiment = True,
                detailed_report=True,
                clip=True,
                positions="/home/jeauscq/Desktop/jcq_thesis/datasets/Testing/test_trajectories_positions.csv",
                )