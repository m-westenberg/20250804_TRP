### /benchmark_stats.py

import json
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Load Data ---

def load_benchmark_results(filepath):
    with open(filepath, 'r') as f:
        return pd.DataFrame(json.load(f))

# --- Compute Mean Metrics Per Algorithm ---

def compute_overall_means(df):
    metrics = ["mean_fitness", "std_fitness", "mean_runtime", "mean_deviation", "mean_overload"]
    summary = df.groupby("algorithm")[metrics].mean().reset_index()
    return summary

# --- Paired Tests for GA vs SA ---

def perform_paired_tests(df, output_dir="paired_ttest_plots"):
    ga_df = df[df['algorithm'] == "GeneticAlgorithm"].sort_values("dataset_id")
    sa_df = df[df['algorithm'] == "SimulatedAnnealing"].sort_values("dataset_id")

    assert len(ga_df) == len(sa_df), "Datasets count mismatch between GA and SA"

    os.makedirs(output_dir, exist_ok=True)
    test_results = {}

    custom_palette = {
        "GeneticAlgorithm": "#e97133",
        "SimulatedAnnealing": "#163e64"
    }

    for metric in ["mean_fitness", "std_fitness", "mean_runtime", "mean_deviation", "mean_overload"]:
        d = ga_df[metric].values - sa_df[metric].values

        # --- Normality Tests ---
        shapiro_stat, shapiro_pval = stats.shapiro(d)
        dagostino_stat, dagostino_pval = stats.normaltest(d)

        normality_result = {
            "shapiro_wilk": {
                "statistic": shapiro_stat,
                "p_value": shapiro_pval,
                "normality": "Fail to reject normality" if shapiro_pval > 0.05 else "Reject normality"
            },
            "dagostino_k2": {
                "statistic": dagostino_stat,
                "p_value": dagostino_pval,
                "normality": "Fail to reject normality" if dagostino_pval > 0.05 else "Reject normality"
            }
        }

        # Dynamically adapt figure size based on data range
        fig_width = 8
        fig_height = 8
        # y_range = max(d) - min(d)
        # if y_range > 5:
            # fig_height = 6

        if metric == "mean_fitness":
            title = "Distribution of Differences in Mean Objective Function Values (GA - SA)"
        elif metric == "std_fitness":
            title = "Distribution of Differences in Mean Standard Deviation Values (GA - SA)"
        elif metric == "mean_runtime":
            title = "Distribution of Differences in Mean Runtime Values (GA - SA)"
        elif metric == "mean_deviation":
            title = f"Distribution of Differences in Mean Deviation (GA - SA)"
        elif metric == "mean_overload":
            title = f"Distribution of Differences in Mean Overload (GA - SA)"
        else:
            title = "idk either what this graph is"

        # Plot histogram of differences with KDE
        plt.figure(figsize=(fig_width, fig_height))
        sns.histplot(d, bins=20, kde=True, color=custom_palette["GeneticAlgorithm"])
        plt.title(title, color="#000000", fontweight='bold')
        plt.xlabel("Difference", color="#000000")
        plt.ylabel("Frequency", color="#000000")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        # pdf_path = os.path.join(output_dir, f"diff_dist_{metric}.pdf")
        png_path = os.path.join(output_dir, f"diff_dist_{metric}.png")
        # plt.savefig(pdf_path, transparent=True)
        plt.savefig(png_path, transparent=True, dpi=300)
        plt.close()

        # Paired t-test
        if metric == "mean_runtime":
            t_stat, t_pval = stats.ttest_rel(ga_df[metric], sa_df[metric], alternative="greater")
        else:
            t_stat, t_pval = stats.ttest_rel(ga_df[metric], sa_df[metric], alternative="less")
        # Wilcoxon signed-rank test
        if metric == "mean_runtime":
            try:
                w_stat, w_pval = stats.wilcoxon(ga_df[metric], sa_df[metric], alternative="greater")
            except ValueError:
                w_stat, w_pval = None, None
        else:
            try:
                w_stat, w_pval = stats.wilcoxon(ga_df[metric], sa_df[metric], alternative="less")
            except ValueError:
                w_stat, w_pval = None, None

        test_results[metric] = {
            "paired_t_test": {
                "t_statistic": t_stat,
                "p_value": t_pval,
                "significance": "Significant" if t_pval < 0.05 else "Not Significant"
            },
            "wilcoxon_signed_rank_test": {
                "w_statistic": w_stat,
                "p_value": w_pval,
                "significance": None if w_pval is None else ("Significant" if w_pval < 0.05 else "Not Significant")
            },
            "normality_test": normality_result
        }

    return test_results

# --- Main ---

def main(filepath="benchmark_results.json", output_file="benchmark_stats.json"):
    df = load_benchmark_results(filepath)
    summary = compute_overall_means(df)
    tests = perform_paired_tests(df)

    output = {
        "overall_means": summary.to_dict(orient="records"),
        "paired_tests": tests
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved benchmark analysis with paired t-test and Wilcoxon signed-rank test to {output_file}")

if __name__ == "__main__":
    main()
