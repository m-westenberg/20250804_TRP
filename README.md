<h1>📌 Project-to-Manager Allocation Optimization</h1>

This repository contains the full implementation, datasets, and visualizations for my **Bachelor Thesis Research Project** on optimizing project-to-manager allocation using **Genetic Algorithms (GA)** and **Simulated Annealing (SA)**.

The project includes:
* **Synthetic dataset generation**
* **Objective function definition** (fitness, deviation, overload)
* **Optimization algorithms** (GA & SA)
* **Benchmarking and statistical comparison**
* **Visualization tools** (heatmaps, workload charts, benchmark plots)
* **Manual allocation evaluation** tools

---

<h2>📂 Repository Structure</h2>

| File | Description |
|------|-------------|
| **`synthetic_dataset_generator.py`** | Generates synthetic datasets of projects & managers with defined characteristics and workload phases. Produces both JSON datasets and PDF reports. |
| **`objective_function.py`** | Defines the objective function (`Z`) and its components (`V` = deviation from contract hours, `P` = overload hours). Includes helper functions for computing workloads. |
| **`genetic_algorithm.py`** | Implements the **Genetic Algorithm** to optimize project-to-manager allocations. |
| **`simulated_annealing.py`** | Implements the **Simulated Annealing** algorithm for the same optimization problem. |
| **`benchmark_algorithms.py`** | Runs both algorithms across all datasets multiple times, measures performance metrics, and stores results in `benchmark_results.json`. |
| **`benchmark_stats.py`** | Performs **paired t-tests**, **Wilcoxon signed-rank tests**, and **normality tests** on benchmark results. Outputs summary statistics and statistical significance analysis. |
| **`visualize_benchmark.py`** | Creates bar plots, boxplots, heatmaps, and scatter plots to visualize benchmark results. |
| **`plot_manager_workload.py`** | Generates heatmaps showing **weekly workload per manager** across the project timeline. Saves all heatmaps to a single PDF. |
| **`custom_allocation.py`** | Allows you to manually define custom projects, projects managers, and a allocation matrix (`x_ij`) of projects to managers and visualize workloads over time. |
| **`ga_allocation.py`** | Allows you to manually define custom projects and project managers, and allocates these projects to the managers using Genetic Algorithm and visualizes workloads over time. |
| **`datasets.json`** | Synthetic datasets generated for optimization runs. |
| **`benchmark_results.json`** | Output file containing algorithm performance results across datasets. |
| **`benchmmark_stats.json`** | Output file containing summary statistics of performanece metrics and statistical significance tests |
| **`visualizations/`** | Contains all generated plots from benchmarking and workload visualizations. |

---

<h2>⚙️ Installation Instructions</h2>

<h3>1️⃣ Install Visual Studio Code</h3>

1. Download **Visual Studio Code**:  
   https://code.visualstudio.com/  
2. Install the **Python extension** in VS Code:  
   * Open VS Code → Go to Extensions (`Ctrl+Shift+X` / `Cmd+Shift+X` on Mac)  
   * Search for **Python** and install the official extension by Microsoft.

---

<h3>2️⃣ Install Python</h3>

1. Download Python from:  
   https://www.python.org/downloads/  
2. During installation:  
   * ✅ **Check “Add Python to PATH”**  
   * Click **Install Now**  

Check if Python is installed:
```bash
python --version
or
python3 --version
```

<h3>3️⃣ Clone This Repository</h3>

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

<h3>4️⃣ Create and Activate a Virtual Environment (Optional but Recommended)</h3>

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

<h3>5️⃣ Install Required Python Libraries</h3>

```bash
pip install numpy pandas matplotlib seaborn scipy
```

<h2>▶️ How to Run</h2> <h3>1. Generate Datasets</h3>

```bash
python data_generator.py
```

<h3>2. Run Optimization</h3>

Genetic Algorithm:
```bash
python genetic_algorithm.py
```

Simulated Annealing:
```bash
python simulated_annealing.py
```

<h3>3. Benchmark Algorithms</h3>

```bash
python benchmark_algorithms.py
```

<h3>4. Analyze Statistics</h3>

```bash
python benchmark_stats.py
```

<h3>5. Visualize Results</h3>

```bash
python visualize_benchmark.py
```

<h3>6. View Manager Workload Heatmaps</h3>

```bash
python plot_manager_workload.py
```

<h3>7. Try Custom Manual Projects, Project Managers, and Allocations</h3>

```bash
python "custom_allocation.py"
```

```bash
python "ga_allocation.py"
```

<h2>📊 Statistical Testing</h2>

The benchmarking analysis:
* Performs paired t-tests to compare GA vs SA.
* Runs Wilcoxon signed-rank tests if normality is rejected.
* Conducts Shapiro–Wilk and D’Agostino’s K² tests for normality.

<h2>📌 Notes</h2>

* All output files (JSON, PDF, PNG) are saved in the repository for reproducibility.
* The analysis assumes each project allocation sums to 100% across managers. The manual allocation script validates this automatically.
* The workload visualizations use actual dates instead of week numbers for easier interpretation.
