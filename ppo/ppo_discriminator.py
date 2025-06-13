# ——— Standard Library ————————————————————————————————————————
import os
import sys
import re
import time
import csv
import random
import logging
from pathlib import Path
from datetime import datetime

# ——— Scientific & RL Libraries ——————————————————————————————
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.envs.registration import register
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

# ——— FLAML ——————————————————————————————————————————————————
from flaml import tune
from flaml.tune import run

# ——— Project Modules ————————————————————————————————————————
current_file = Path().resolve()
while current_file.name != "jcq_thesis" and current_file != current_file.parent:
    current_file = current_file.parent
sys.path.append(str(current_file))

from utils.config import BASE_DIR
from ppo.envs.CustomArmDiscr import Custom2DoFEnv
from ppo.utils.agent import Agent
from ppo.utils.discriminator import EnsembleDiscriminator, train_discriminator_ensemble
from ppo.evaluate_ppo import evaluate_policy

# ——— Directories ————————————————————————————————————————————
DATASET_DIR = BASE_DIR + "datasets/"
# EXISTING_EXPERIMENT = BASE_DIR + "ppo/best_experiments/trial_20250506_172040/"

EXISTING_EXPERIMENT = "/home/jeauscq/Desktop/RESULTS/27Mayo/experiment033/runs/trial_20250526_195649/"
os.makedirs(BASE_DIR + "ppo/experiments/", exist_ok=True)
existing = [d for d in os.listdir(BASE_DIR + "ppo/experiments/") if re.match(r"experiment\d{3}", d)]
nums = [int(re.findall(r"\d{3}", d)[0]) for d in existing] if existing else [0]
CURRENT_EXP_DIR = os.path.join(BASE_DIR + "ppo/experiments/", f"experiment{(max(nums) + 1):03d}")
os.makedirs(CURRENT_EXP_DIR, exist_ok=True) # Creates directory ppo/experiments/experimentXXX

TRAINING_DATASET = DATASET_DIR + "Policy/MPC/torque_Const/2iteration/mpc_generated_tor_2_constrained_dataset.csv"
TRAINING_DATASET_N_STATS = DATASET_DIR + "Policy/MPC/torque_Const/2iteration/mpc_generated_tor_2_constrained_dataset_n_stats.txt"

EXPERT_DATASET_N = DATASET_DIR + "Discriminator/MPC/torque_Const/mpc_generated_tor_constraint_expert_dataset_n.csv"

TEST_DATASET = DATASET_DIR + "Testing/test_trajectories.csv"
TEST_DATASET_N_STATS = DATASET_DIR + "Testing/test_trajectories_n_stats.txt"
# ——— Flags —————————————————————————————————————————————————
TUNING = False                # It uses flaml to find the best hyperparameters
FINE_TUNING = False           # It loads an existing model (EXISTING_EXPERIMENT) and fine-tunes it
RESET_OPTIMIZER = True        # It resets the optimizer to a new learning rate (Fine-tuning)
CONSTRAINT_ANALYSIS = True    # It defines whether it checks all the possibilities of constraint enforcing

# ——— Constants ——————————————————————————————————————————————
DIM_SHAPE = 4

# ——— Logging ————————————————————————————————————————————————
log_file = os.path.join(CURRENT_EXP_DIR, f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
sys.stdout = sys.stderr = open(log_file, "w", encoding="utf-8")


def load_hyperparameters(run_path):
    """
    Loads hyperparameters from a TensorBoard event file.
    Args:
        run_path (str): Path to the TensorBoard event file.
    Returns:
        hyperparams (dict): Dictionary of hyperparameters.
    """
    # Load the event file
    ea = event_accumulator.EventAccumulator(run_path)
    ea.Reload()

    # Extract hyperparameters from text (tensor) entry
    try:
        text_events = ea.Tensors("hyperparameters/text_summary")
        if text_events:
            text_val = text_events[-1].tensor_proto.string_val[0].decode("utf-8")
            hyperparam_lines = text_val.strip().split("\n")[2:]  # skip markdown headers
            hyperparams = {line.split("|")[1]: line.split("|")[2] for line in hyperparam_lines if "|" in line}
        else:
            hyperparams = {}
    except Exception as e:
        hyperparams = {}
        print(f"Failed to parse hyperparameters from {run_path}: {e}")
    return hyperparams


def load_trajectories(csv_path,):
    """
    Loads all trajectories from a CSV file.
    Args:
        csv_path (str): Path to CSV.
        dim_shape (int, optional): State dimension (e.g., 4).
        traj_len (int, optional): Expected trajectory length (e.g., 500).
        as_numpy (bool): Whether to return NumPy arrays.
    Returns:
        List of trajectories (either list of lists or list of NumPy arrays).
    """
    trajectories = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            trajectories.append(np.array([list(map(float, state.split(";"))) for state in row]))
    return trajectories


def sample_structured_obs(trajectories, N, M, num_samples):
    """
    Samples structured observations from trajectories.
    Args:
        trajectories (list): List of trajectories.
        N (int): Number of past states.
        M (int): Number of future desired states.
        num_samples (int): Number of samples to generate.
    Returns:
        structured_obs (np.ndarray): Array of structured observations.
    """
    structured_obs = []
    for _ in range(num_samples):
        traj = random.choice(trajectories)
        t = random.randint(N, len(traj) - M - 1)
        window = traj[t - N : t + M + 1]
        structured_obs.append(window)
    return np.stack(structured_obs)


def make_env(env_id, M, N, gamma, discriminator, traj, normalize_stats_path=None, rewardMode=0):
    """
    Creates a gym environment with the specified parameters.
    Args:
        env_id (str): Environment ID.
        M (int): Number of future desired states.
        N (int): Number of past states.
        gamma (float): Discount factor.
        discriminator (EnsembleDiscriminator): Discriminator object.
        traj (list): Trajectory data.
        normalize_stats_path (str, optional): Path to normalization statistics file.
    Returns:
        thunk (function): Function to create the environment.
    """
    def thunk():
        env = gym.make(env_id, trajectories=traj, discriminator=discriminator, M=M, N=N, normalize_path=normalize_stats_path, rewardMode=rewardMode)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if rewardMode > 1:
            env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -100, 100)) # Could be removed eventually I think
        return env
    return thunk


def training_loop(config, total_timesteps=6e6, rewardMode=0):

    # Cuda and torch
    import torch
    import torch.nn as nn
    import torch.optim as optim
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Cuda version: {torch.version.cuda}")
    print(f"Torch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_deterministic = True
    print(f"Using device: {device}")
    
    # Seeding
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    best_loss = 2

    if FINE_TUNING:
        # Keep the following hyperparameters from the original experiment, in case we are fine-tuning.
        keep_hyperparameters = ["hidden_dim", "num_heads", "N", "M"]
        index_keys = load_hyperparameters(EXISTING_EXPERIMENT)
        config.update({k: int(index_keys[k]) for k in keep_hyperparameters})
        print(f'Loaded hidden_dim is: {config["hidden_dim"]}')


    # Initialize the agent and discriminator
    discriminator = EnsembleDiscriminator(input_dim=DIM_SHAPE, hidden_dim=config["hidden_dim"], num_heads=config["num_heads"]).to(device)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=config["lr_disc"])
    agent = Agent(M=config["M"], N=config["N"], hidden_dim=config["hidden_dim"], std_init=list([config["std_init"], config["std_init"]/2])).to(device)
    optimizer = optim.Adam([{"params": agent.actor_parameters(), "lr": config["lr_pol"]}, {"params": agent.critic_parameters(), "lr": config["lr_val"]},], eps=1e-5)

    if FINE_TUNING:
        # Load the checkpoint from file
        checkpoint = torch.load(f"{EXISTING_EXPERIMENT}/models_and_optimizers.pt", map_location=device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        print("Agent loaded from checkpoint")
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        print("Discriminator loaded from checkpoint")
        if not RESET_OPTIMIZER:
            # This will load the optimizer state dicts including the original learning rates
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])

    # Update config with extra static parameters
    config.update({
        "torch_deterministic": torch_deterministic,                             # Whether to make PyTorch operations deterministic
        "gae_lambda": 0.95,                                                     # Lambda parameter for Generalized Advantage Estimation (GAE)
        "clip_coef": 0.2,                                                       # PPO clip coefficient for policy update
        "clip_vloss": True,                                                     # Whether to clip value loss updates
        "vf_coef": 0.5,                                                         # Coefficient for value function loss in PPO objective
        "max_grad_norm": 0.5,                                                   # Maximum gradient norm for clipping (to avoid explosion)
        "target_kl": None,                                                      # Target KL divergence for early stopping PPO updates
        "norm_adv": True,                                                       # Whether to normalize advantages before PPO update
        "env_id": "Custom2DoF-v0",                                              # Environment ID name 
        "seed": seed,                                                           # Seed for reproducibility
        "num_params_policy": sum(p.numel() for p in agent.parameters()),        # Number of trainable parameters of the policy
        "num_params_disc": sum(p.numel() for p in discriminator.parameters()),  # Number of trainable parameters of the discriminator
        "rewardMode": rewardMode,                                               # Indicates the constraint analysis mode. 0: Nothing, 1: Reward Penalization, 2: 1+Clipping, 3: Only clipping
        })

    register(id="Custom2DoF-v0", entry_point="ppo.envs.CustomArmDiscr:Custom2DoFEnv",)  # "module_path:ClassName"

    # Sizes
    num_steps = int(config["batch_size"] // config["num_envs"])                   # Number of steps per environment per iteration
    minibatch_size = int(config["batch_size"] // config["num_minibatches"])       # Minibatch size according to the number of minibatches
    num_iterations = int(total_timesteps // config["batch_size"])                      # The total number of iterations that will be required to have the total_timesteps
    eval_frequency_iter = 10                                                      # Defines how often the loss is reported to the optimization algorithm
    print_freq = num_steps//50                                                    # Frequency of printed updates
    num_traj_test = 60

    # Storage name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")                                # Format as string: YYYYMMDD_HHMMSS
    run_name = f"trial_{str(timestamp)}"                                                # Name of folder inside of run 
    writer = SummaryWriter(CURRENT_EXP_DIR + f"/runs/{run_name}")                       # Initializes writer
    checkpoint_path = CURRENT_EXP_DIR + f"/runs/{run_name}/models_and_optimizers.pt"    # Name of the model

    # Reports hyperparameters
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),)

    # Load trajectories
    traj = load_trajectories(TRAINING_DATASET)
    expert_trajectories = load_trajectories(EXPERT_DATASET_N)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(env_id=config["env_id"], M=config["M"], N=config["N"], gamma=config["gamma"], discriminator=discriminator, traj=traj,
                                              normalize_stats_path=TRAINING_DATASET_N_STATS, rewardMode=config["rewardMode"]) for _ in range(int(config["num_envs"]))])

    # Storage setup
    obs = torch.zeros((num_steps, config["num_envs"]) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, config["num_envs"]) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, config["num_envs"])).to(device)
    rewards = torch.zeros((num_steps, config["num_envs"])).to(device)
    dones = torch.zeros((num_steps, config["num_envs"])).to(device)
    values = torch.zeros((num_steps, config["num_envs"])).to(device)
    gen_obs_windows = []

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config["num_envs"]).to(device)

    if TUNING and FINE_TUNING:
        pos_rmse, vel_rmse, success_rate = evaluate_policy(config=config, model_path="EarlyStopping Check", trajectories_path=TEST_DATASET,
                                            base_dir= CURRENT_EXP_DIR, final=False, run_name=run_name, step=str(global_step), device=device,
                                            num_traj=num_traj_test, policy=agent, normalize_stats_path=TEST_DATASET_N_STATS, clip=rewardMode>1)
        best_loss = pos_rmse + vel_rmse
        tune.report({"loss": pos_rmse + vel_rmse, "step": global_step, })
        writer.add_scalar("losses/test_loss", pos_rmse+vel_rmse, global_step)
        writer.add_scalar("losses/success_rate_test", success_rate, global_step)
        print(f"Reported loss {pos_rmse + vel_rmse} at step {global_step}, for run {run_name}")
    for iteration in range(1, num_iterations + 1):
        # Annealing the learning rate
        if config["anneal_lr"]:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            # Anneal Policy's learning rate
            lrnow = frac * config["lr_pol"]
            optimizer.param_groups[0]["lr"] = lrnow
            # Anneal Discriminator's learning rate
            lrnow = frac * config["lr_disc"]
            disc_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += config["num_envs"]

            # The current state is the next state of the past step
            obs[step] = next_obs
            dones[step] = next_done

            # -- ACTION LOGIC --
            with torch.no_grad():
                reshaped_obs = next_obs.view(config["num_envs"], config["N"]+1+config["M"], DIM_SHAPE)
                # For the disc we only store obs that contain actual past and current states
                if step > config["N"]:
                    gen_obs_windows.append(reshaped_obs.clone().detach().cpu())
                # Predicts action and value for current state
                action, logprob, _, value = agent.get_action_and_value(reshaped_obs.to(device))

		        # Store in buffers
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # -- EXECUTE THE GAME AND LOG DATA --
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        if  global_step>print_freq:
                            print_freq += print_freq
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

        # Control stats calcualted here for iteration
        delta_torque = (actions[1:] - actions[:-1]).norm(dim=-1)
        mean_delta_torque_norm = delta_torque.mean().item()
        mean_torque_norm = actions.norm(dim=-1).mean().item()

        # -- TRAINING DISCRIMINATOR --
        gen_obs_all = torch.cat(gen_obs_windows, dim=0).detach()
        sample_size = min(len(gen_obs_all), config["batch_disc"])
        expert_obs_batch = sample_structured_obs(expert_trajectories, config["N"], config["M"], sample_size)  # Expert trajectories
        if len(gen_obs_all) > sample_size:
            idx = np.random.choice(np.arange(len(gen_obs_all)), size=sample_size, replace=False)
            gen_obs_all = gen_obs_all[idx]  # Randomly sample from generated observations
        disc_loss = train_discriminator_ensemble(discriminator, disc_optimizer, expert_obs_batch, gen_obs_all, lambda_gp=config["lambda_gp"], batch_size=config["minibatch_disc"], device=device)  # Trains discriminator
        gen_obs_windows.clear()
        disc_rewards_all = rewards.view(-1)

        # bootstrap value if not done
        with torch.no_grad():
            reshaped_obs = next_obs.view(config["num_envs"], config["N"]+config["M"]+1, DIM_SHAPE)
            next_value = agent.get_value(reshaped_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config["gamma"] * config["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_obs = b_obs.view(-1, config["N"]+config["M"]+1, DIM_SHAPE)  # Reshape flat obs to (batch, seq_len, 4)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # --TRAINING POLICY AND VALUE NETWORK --
        b_inds = np.arange(config["batch_size"])
        clipfracs = []
        for epoch in range(config["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, config["batch_size"], minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config["clip_coef"]).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["clip_coef"], 1 + config["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config["clip_vloss"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -config["clip_coef"], config["clip_coef"],)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config["ent_coef"] * entropy_loss + v_loss * config["vf_coef"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"])
                optimizer.step()

            if config["target_kl"] is not None and approx_kl > config["target_kl"]:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        with torch.no_grad():
            action_std = agent.actor_logstd.exp().mean().item()

        # -- REPORTING --
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("losses/discriminator", disc_loss, global_step)
        writer.add_scalar("statistics/discriminator_reward_mean", disc_rewards_all.mean().item(), global_step)
        writer.add_scalar("statistics/discriminator_reward_std", disc_rewards_all.std().item(), global_step)
        writer.add_scalar("statistics/policy_action_std", action_std, global_step)
        writer.add_scalar("statistics/mean_torque_norm", mean_torque_norm, global_step)
        writer.add_scalar("statistics/mean_delta_torque_norm", mean_delta_torque_norm, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        print(f"Iteration: {iteration}")


        if iteration % (eval_frequency_iter) == 0:
            pos_rmse, vel_rmse, success_rate = evaluate_policy(config=config, model_path="EarlyStopping Check", trajectories_path=TEST_DATASET,
                                                 base_dir= CURRENT_EXP_DIR, final=False, run_name=run_name, step=str(global_step), device=device,
                                                 num_traj=num_traj_test, policy=agent, normalize_stats_path=TEST_DATASET_N_STATS, clip=rewardMode>1)
            writer.add_scalar("losses/test_loss", pos_rmse+vel_rmse, global_step)
            writer.add_scalar("losses/success_rate_test", success_rate, global_step)
            print(f"Pos loss: {pos_rmse} Vel loss: {vel_rmse} at step {global_step}, for run {run_name}")

            if pos_rmse+vel_rmse < best_loss:
                best_loss = pos_rmse+vel_rmse
                checkpoint = {"agent_state_dict": agent.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                                "discriminator_state_dict": discriminator.state_dict(), "disc_optimizer_state_dict": disc_optimizer.state_dict(),}
                torch.save(checkpoint, checkpoint_path[:-3]+"_int.pt")
                print(f"Intermediate checkpoint saved to {checkpoint_path}")

            if TUNING:
                tune.report({"loss": pos_rmse + vel_rmse, "step": global_step, })

    checkpoint = {"agent_state_dict": agent.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                  "discriminator_state_dict": discriminator.state_dict(), "disc_optimizer_state_dict": disc_optimizer.state_dict(),}

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    envs.close()
    writer.close()

    # -- FINAL EVALUATION --
    pos_rmse, vel_rmse, _ = evaluate_policy(config=config, model_path=checkpoint_path, trajectories_path=TEST_DATASET, 
                                         base_dir=CURRENT_EXP_DIR, final=True, run_name=run_name, device=device, num_traj=200,
                                         normalize_stats_path=TEST_DATASET_N_STATS, clip=rewardMode>1)
    return {"metric/loss": pos_rmse + vel_rmse}


def objective(config):
    result = training_loop(config)
    return result

def trial_name_creator(trial):
    return f"trial_{trial.trial_id[:6]}"  # or any short string you prefer

def trial_dirname_creator(trial):
    return f"trial_{trial.trial_id[:6]}"    

if __name__ == "__main__":
    if TUNING:
        search_space = {
            "anneal_lr": tune.choice([True, False]),            # Whether to decay learning rate during training 
            "ent_coef": tune.loguniform(1e-9, 1e-3),            # Entropy coefficient to encourage exploration
            "batch_size": tune.choice([2048, 4096, 8192]),      # Number of collected samples per PPO rollout batch
            "num_minibatches": tune.choice([16, 32, 64, 128]),  # Number of minibatches to split batch_size during optimization
            "num_envs": tune.choice([4, 8, 16]),                # Number of parallel environments for faster data collection
            "update_epochs": tune.randint(4, 10),               # Number of optimization epochs per PPO update
            "hidden_dim": tune.choice([48, 96, 128, 256]),      # Hidden layer size for the policy, value and discriminator networks
            "num_heads": tune.choice([8, 16, 32]),              # Number of heads (for multi-head discriminators ensemble) 
            "lambda_gp": tune.randint(6, 15),                   # Gradient penalty coefficient for discriminator regularization
            "batch_disc": tune.choice([4096, 8192, 16384]),     # Size of discriminator training batch for real and generated samples
            "minibatch_disc": tune.choice([256, 512, 1024]),    # Number of minibatches for discriminator update
            "std_init": tune.uniform(0.001, 3.0),               # Initial standard deviations for policy output (2 values)
            "lr_pol": tune.loguniform(5e-5, 5e-3),              # Learning rate for the policy network
            "lr_val": tune.loguniform(5e-5, 1e-3),              # Learning rate for the value network
            "lr_disc": tune.loguniform(5e-5, 1e-3),             # Learning rate for the discriminator network
            "gamma": tune.choice([0.95, 0.975, 0.99]),          # Discount factor for future rewards
            "N": tune.choice([3, 5]),                           # Number of past states included in observation
            "M": tune.choice([11, 12]),                         # Number of future desired states included in observation
        }
        
        analysis = run(objective, config=search_space, metric="metric/loss", mode="min", num_samples=50 , use_ray=True, trial_name_creator=trial_name_creator,
                    trial_dirname_creator=trial_dirname_creator, resources_per_trial={"gpu": 0.3333, "cpu":12}, search_alg="BlendSearch", 
                       n_concurrent_trials=4, scheduler="asha", resource_attr="metric/step", min_resource=1000000, max_resource=6000000) 

        # Extract all results
        results = []
        for trial in analysis.trials:
            entry = {"trial_id": trial.trial_id}

            # Flatten config (hyperparameters)
            for k, v in trial.config.items():
                entry[f"config/{k}"] = v

            # Flatten metrics (final results)
            if hasattr(trial, "last_result") and trial.last_result:
                for k, v in trial.last_result.items():
                    if "config" in k:
                        continue # Skip any config keys mistakenly logged in metrics
                    entry[f"metric/{k}"] = v

            results.append(entry)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Print and save clean CSV
        print("\nAll trial results:")
        print(df.head())

        csv_name = BASE_DIR + "ppo/flaml_results.csv"
        df.to_csv(csv_name, index=False, sep=";")
        print(f"Clean results saved to '{csv_name}'")

        # Get best result manually
        best_config = analysis.best_config
        best_loss = analysis.best_trial.last_result["loss"]

        print("\nBest config found:")
        print(best_config)
        print(f"Best loss achieved: {best_loss}")
    else:
        config = {'anneal_lr': False, 'ent_coef': 0.0001815466939005705, 'batch_size': 2048, 'num_minibatches': 64,
                  'num_envs': 16, 'update_epochs': 7, 'hidden_dim': 256, 'num_heads': 8, 'lambda_gp': 8, 'batch_disc': 4096,
                  'minibatch_disc': 1024, 'std_init': 3.0, 'lr_pol': 5e-05, 'lr_val': 0.00044283380419313267, 'lr_disc': 4.1158243033544476e-05,
                  'gamma': 0.975, 'N': 5, 'M': 11, 'gae_lambda': 0.95, 'clip_coef': 0.2, 'clip_vloss': True, 'vf_coef': 0.5, 'max_grad_norm': 0.5,
                  'target_kl': None, 'norm_adv': True, 'env_id': 'Custom2DoF-v0',}
        
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
        
        timesteps = 20000000
        if CONSTRAINT_ANALYSIS:
            rewardModes = [1, 2, 3] # 0-> Nothing, 1-> Penalize high torques, 2->Penalize high torques and clip them, 3->Only clip them
        else:
            rewardModes = [0]
        
        for rewardMode in rewardModes:
            training_loop(config, total_timesteps=timesteps, rewardMode=rewardMode)

    sys.stdout.close()
    sys.stdout = sys.__stdout__  # Reset back to normal terminal printing
