### /visualize_benpmarks.py 
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Load Benchmark Results ---
def load_results(filepath="benchmark_results.json"):
    with open(filepath, "r") as f:
        return pd.DataFrame(json.load(f))

# --- Common Colors ---
CUSTOM_COLORS = {
    "GeneticAlgorithm": "#e97133",  # Orange
    "SimulatedAnnealing": "#163e64"  # Dark blue
}
TEXT_COLOR = "#000000"

# --- Box Plot for Each Metric ---
def plot_boxplot(df, metric, ylabel, save_dir, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.boxplot(
        data=df,
        x="algorithm",
        y=metric,
        hue="algorithm",
        dodge=False,
        palette=CUSTOM_COLORS,
        ax=ax,
        legend=False
    )
    
    # Add stripplot overlay for distribution points
    # sns.stripplot(data=df, x="algorithm", y=metric, hue="algorithm", dodge=False, palette=CUSTOM_COLORS, size=3, alpha=0.5, ax=ax, legend=False)

    # Add numerical annotations for quartiles
    for i, algo in enumerate(df['algorithm'].unique()):
        vals = df[df['algorithm'] == algo][metric]
        q1 = np.percentile(vals, 25)
        median = np.median(vals)
        q3 = np.percentile(vals, 75)
        ax.text(i, median, f"{median:.1f}", ha='center', va='center', color='w', fontweight='bold', fontsize=8)
        # ax.text(i, q1, f"{q1:.1f}", ha='center', va='top', color='w', fontweight='bold', fontsize=8)
        # ax.text(i, q3, f"{q3:.1f}", ha='center', va='bottom', color='w', fontweight='bold', fontsize=8)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.set_xlabel('', color=TEXT_COLOR)
    ax.set_title(f"{title}", color=TEXT_COLOR, fontweight='bold')
    ax.tick_params(colors=TEXT_COLOR)
    filename_base = metric.replace(" ", "_").lower()
    png_path = os.path.join(save_dir, f"boxplot_{filename_base}.png")
    plt.savefig(png_path, transparent=True, dpi=300)
    plt.close()

# --- Strip Plot for Each Metric ---
def plot_stripplot(df, metric, ylabel, save_dir, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.stripplot(
        data=df,
        x="algorithm",
        y=metric,
        hue="algorithm",
        dodge=False,
        palette=CUSTOM_COLORS,
        size=4,
        alpha=0.7,
        ax=ax,
        legend=False
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.set_xlabel('', color=TEXT_COLOR)
    ax.set_title(f"{title}", color=TEXT_COLOR, fontweight='bold')
    ax.tick_params(colors=TEXT_COLOR)
    filename_base = metric.replace(" ", "_").lower()
    png_path = os.path.join(save_dir, f"stripplot_{filename_base}.png")
    plt.savefig(png_path, transparent=True, dpi=300)
    plt.close()

# --- Main Visualization Entry ---
def visualize_summary(output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    df = load_results()
    
    metrics = [
        ("mean_fitness", "Objective Function Values", "Solution Quality"),
        ("std_fitness", "Standard Deviation Values", "Robustness"),
        ("mean_runtime", "Runtime Values (in seconds)", "Computational Efficiency"),
        ("mean_deviation", "Deviation", "Deviation from Contract Hours"),
        ("mean_overload", "Overload", "Overload Hours")
    ]
    
    for metric, ylabel, title in metrics:
        plot_boxplot(df, metric, ylabel, output_dir, title)
        plot_stripplot(df, metric, ylabel, output_dir, title)
    
    print(f"Saved individual box plots and strip plots for each metric to {output_dir}/ as transparent PNGs")

if __name__ == "__main__":
    visualize_summary()
