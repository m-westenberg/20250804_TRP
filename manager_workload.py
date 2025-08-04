### /plot_manager_workload.py

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from objective_function import compute_all_workloads, compute_W

# --- Load Benchmark and Dataset Data ---

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# --- Heatmap of Manager Workload Over Time ---

def heatmap_manager_workload(ax, x_ij, projects, managers, weeks, dataset_id, algorithm):
    project_workloads = compute_all_workloads(projects, weeks)
    W = compute_W(x_ij, project_workloads, managers)  # shape: (num_managers, weeks)

    W_rounded = np.round(W).astype(int)
    manager_labels = [f"M{i+1}\n({managers[i]['contract_hours']}h)" for i in range(W.shape[0])]
    sns = __import__('seaborn')
    sns.heatmap(W, cmap="YlOrRd", cbar=True, ax=ax,
                xticklabels=1, yticklabels=manager_labels,
                annot=W_rounded, fmt=".2f", annot_kws={"size": 10})
    ax.set_title(f"Manager Workload Heatmap - Dataset {dataset_id} ({algorithm})")
    ax.set_xlabel("Week")
    ax.set_ylabel("Manager")

# --- Main Entry ---

def main(benchmark_file, dataset_file, output_dir="manager_workload_plots"):
    results = load_json(benchmark_file)
    datasets = load_json(dataset_file)
    dataset_map = {d['dataset_id']: d for d in datasets}

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "manager_workloads_all.pdf")

    with PdfPages(pdf_path) as pdf:
        for entry in results:
            x_ij = entry.get("best_solution")
            if x_ij is None:
                continue

            dataset_id = entry['dataset_id']
            dataset = dataset_map.get(dataset_id)
            if not dataset:
                continue

            fig_width = max(16, dataset['weeks'])
            fig, ax = plt.subplots(figsize=(fig_width, 6))
            heatmap_manager_workload(
                ax,
                x_ij,
                dataset['projects'],
                dataset['project_managers'],
                dataset['weeks'],
                dataset_id,
                entry['algorithm']
            )
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved all manager workload heatmaps to {pdf_path}")

if __name__ == "__main__":
    main("benchmark_results.json", "datasets.json")