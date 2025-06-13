import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_action_bounds(csv_path, t1min, t1max, t2min, t2max, output_path="action_histograms.pdf"):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Identify out-of-bounds for each action
    out_1 = (df['action_1'] < t1min) | (df['action_1'] > t1max)
    out_2 = (df['action_2'] < t2min) | (df['action_2'] > t2max)
    total_out = out_1 | out_2

    # Statistics
    total_steps = len(df)
    num_out_of_bounds = total_out.sum()
    percent_out = (num_out_of_bounds / total_steps) * 100

    print(f"Total steps: {total_steps}")
    print(f"Out-of-bounds steps: {num_out_of_bounds}")
    print(f"Percentage of out-of-bounds actions: {percent_out:.2f}%")

    # Plot histograms
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot for action_1
    sns.histplot(df['action_1'], bins=50, ax=axs[0], kde=False, color='skyblue')
    axs[0].axvspan(t1min, t1max, color='lightgreen', alpha=0.3, label='In bounds')
    axs[0].axvline(t1min, color='red', linestyle='--', linewidth=1)
    axs[0].axvline(t1max, color='red', linestyle='--', linewidth=1)
    axs[0].set_title("Action 1 Histogram")
    axs[0].set_xlabel("action_1")
    axs[0].set_ylabel("Count")

    # Plot for action_2
    sns.histplot(df['action_2'], bins=50, ax=axs[1], kde=False, color='skyblue')
    axs[1].axvspan(t2min, t2max, color='lightgreen', alpha=0.3, label='In bounds')
    axs[1].axvline(t2min, color='red', linestyle='--', linewidth=1)
    axs[1].axvline(t2max, color='red', linestyle='--', linewidth=1)
    axs[1].set_title("Action 2 Histogram")
    axs[1].set_xlabel("action_2")
    axs[1].set_ylabel("Count")

    # Shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    axs[1].legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved histogram to {output_path}")

def analyze_action_bounds_2(csv_path, t1min, t1max, t2min, t2max, output_path="action_histograms.pdf"):
    # Load and parse the dataset
    with open(csv_path, "r") as file:
        lines = file.readlines()

    actions_t1 = []
    actions_t2 = []

    for line in lines:
        states = line.strip().split(',')
        for state in states:
            try:
                t1, t2 = map(float, state.split(';'))
                actions_t1.append(t1)
                actions_t2.append(t2)
            except ValueError:
                continue  # Skip malformed entries

    df = pd.DataFrame({"action_1": actions_t1, "action_2": actions_t2})

    # Identify out-of-bounds for each action
    out_1 = (df['action_1'] < t1min) | (df['action_1'] > t1max)
    out_2 = (df['action_2'] < t2min) | (df['action_2'] > t2max)
    total_out = out_1 | out_2

    # Statistics
    total_steps = len(df)
    num_out_of_bounds = total_out.sum()
    percent_out = (num_out_of_bounds / total_steps) * 100

    print(f"Total steps: {total_steps}")
    print(f"Out-of-bounds steps: {num_out_of_bounds}")
    print(f"Percentage of out-of-bounds actions: {percent_out:.2f}%")

    # Plot histograms
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot for action_1
    sns.histplot(df['action_1'], bins=15, ax=axs[0], kde=False, color='skyblue')
    axs[0].axvspan(t1min, t1max, color='lightgreen', alpha=0.3, label='In bounds')
    axs[0].axvline(t1min, color='red', linestyle='--', linewidth=1)
    axs[0].axvline(t1max, color='red', linestyle='--', linewidth=1)
    axs[0].set_title("Action 1 Histogram")
    axs[0].set_xlabel("action_1")
    axs[0].set_ylabel("Count")
    axs[0].set_xlim(-100, 100)


    # Plot for action_2
    sns.histplot(df['action_2'], bins=15, ax=axs[1], kde=False, color='skyblue')
    axs[1].axvspan(t2min, t2max, color='lightgreen', alpha=0.3, label='In bounds')
    axs[1].axvline(t2min, color='red', linestyle='--', linewidth=1)
    axs[1].axvline(t2max, color='red', linestyle='--', linewidth=1)
    axs[1].set_title("Action 2 Histogram")
    axs[1].set_xlabel("action_2")
    axs[1].set_ylabel("Count")

    # Shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    axs[1].legend(handles, labels, loc='upper right')
    axs[1].set_xlim(-50, 50)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved histogram to {output_path}")


if __name__ == "__main__":
    rewardMode = 2
    interm = "_int"
    processor = "cpu" # cpu - cuda
    soft = False
    if soft:
        constraint = "softer"
        t1min=-38.0
        t1max=33.0   
        t2min=-18.0
        t2max=17.0
    else:
        constraint = "stricter"
        t1min=-20.000000000000002
        t1max=18.0000000000000002
        t2min=-12.00000000000002
        t2max=12.000000000000002

    analyze_action_bounds(
        csv_path=f'/home/jeauscq/Desktop/ResultsThesis/3.Const/rewardMode{rewardMode}/S{constraint[1:]}/time_experiment/report{interm}_{processor}/_detailed_actions.csv',
        # csv_path="/home/jeauscq/Desktop/ResultsThesis/2.Unconst/30M/time_experiment/full_report_taskset_cpu/_detailed_actions.csv",
        t1min=t1min, t1max=t1max,
        t2min=t2min, t2max=t2max,
        output_path=f"/home/jeauscq/Desktop/ResultsThesis/3.Const/rewardMode{rewardMode}/S{constraint[1:]}/rewardMode{rewardMode}_{constraint}_action{interm}_{processor}_histograms.pdf"
        # output_path="/home/jeauscq/Desktop/lol.pdf"
    )
