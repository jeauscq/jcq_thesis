import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_combined_actions(csv_paths, labels):
    """Load actions and label them per configuration."""
    all_data = []
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        df["Constraints Handling"] = label
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def plot_action_histogram(df, action_col, tmin, tmax, output_path):
    """Plot a single histogram comparing multiple configurations for one joint."""
    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x=action_col,
        hue="Constraints Handling",
        bins=30,
        element="step",
        stat="count",
        # common_norm=False,
        palette="colorblind",
        multiple="layer",
        kde=False,
        alpha=0.60   ,
        linewidth=0.5,
        edgecolor="black"
    )
    plt.axvline(tmin, color='red', linestyle='--', linewidth=1)
    plt.axvline(tmax, color='red', linestyle='--', linewidth=1)
    plt.axvspan(tmin, tmax, color='lightgreen', alpha=0.15)
    plt.title(f"$\\tau_{{{action_col[-1]}}}$ Distribution Across Configurations")
    handles, labels = plt.gca().get_legend_handles_labels()

    # plt.legend(labels=["Penalty + Clipping", "Clipping"],title="Constraints Handling", loc="upper center", ncol=2)


    plt.xlabel("Torque (Nm)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved histogram for {action_col} to {output_path}")

if __name__ == "__main__":
    # Configuration
    constraint = "stricter"
    processor = "cpu"
    interm = "_int"
    base_path = "/home/jeauscq/Desktop/ResultsThesis/3.Const"

    csv_paths = [
        f"{base_path}/rewardMode0/S{constraint[1:]}/time_experiment/report{interm}_{processor}/_detailed_actions.csv",
        f"{base_path}/rewardMode1/S{constraint[1:]}/time_experiment/report{interm}_{processor}/_detailed_actions.csv",
        # f"{base_path}/rewardMode3/S{constraint[1:]}/time_experiment/report{interm}_{processor}/_detailed_actions.csv",
        # f"{base_path}/rewardMode2/S{constraint[1:]}/time_experiment/report{interm}_{processor}/_detailed_actions.csv"
    ]

    labels = [
        "Constrained Expert Dataset",
        "Penalty",
        # "Clipping",
        # "Penalty + Clipping"
    ]

    # Joint torque bounds for "softer" case
    t1min, t1max = -20, 18
    t2min, t2max = -12, 12

    # Load all actions
    df = load_combined_actions(csv_paths, labels)

    # Plot separately for each joint
    plot_action_histogram(
        df=df,
        action_col="action_1",
        tmin=t1min,
        tmax=t1max,
        output_path=f"{base_path}/combined_action1_hist_{constraint}_{processor}.pdf"
    )

    plot_action_histogram(
        df=df,
        action_col="action_2",
        tmin=t2min,
        tmax=t2max,
        output_path=f"{base_path}/combined_action2_hist_{constraint}_{processor}.pdf"
    )
