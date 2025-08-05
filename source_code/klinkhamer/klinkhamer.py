import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages

# -------------------------
# Helper functions for phase calculations
# -------------------------

def generate_phase_windows_and_workload(size, duration, start_week, base_phase_weights, base_phase_durations):
    phase_weights = np.array(base_phase_weights) / sum(base_phase_weights)

    variable_duration = duration - 1
    phase_durations = []
    for i, frac in enumerate(base_phase_durations):
        if i == 4:
            phase_durations.append(1)
        else:
            phase_durations.append(int(round(frac * variable_duration)))

    while sum(phase_durations) != duration:
        diff = duration - sum(phase_durations)
        phase_durations[0] += diff

    pointer = start_week
    phase_starts = []
    for dur in phase_durations:
        phase_starts.append(pointer)
        pointer += dur
    phase_windows = [(start, start + dur - 1) for start, dur in zip(phase_starts, phase_durations)]

    phase_workload = [float(size * w) for w in phase_weights]
    return phase_windows, phase_workload

# -------------------------
# Objective Function
# -------------------------
# Checks if assignment matrix x_ij is feasible:
# Each project (column) must have total assignment sum ≈ 1 (fully assigned)
def is_feasible(x_ij):
    x = np.array(x_ij)
    project_sums = np.sum(x, axis=0)
    return np.allclose(project_sums, 1.0, atol=1e-3)

# Ensures all assignment values are clipped to [0,1] range (probabilistic interpretation)
def clip_xij(x_ij):
    return np.clip(x_ij, 0.0, 1.0)

# Computes the weekly workload of a single project over the time horizon
# Sums evenly-distributed phase workloads into a time-aligned array
def compute_project_workload(project, time_horizon):
    workload = np.zeros(time_horizon)
    for (start, end), hours in zip(project['phase_windows'], project['phase_workload']):
        duration = end - start + 1
        if duration > 0:
            weekly_hours = hours / duration
            for t in range(start, end + 1):
                if 0 <= t < time_horizon:
                    workload[t] += weekly_hours
    return workload

# Computes the workload for all projects as a list of time series vectors
def compute_all_workloads(projects, time_horizon):
    return [compute_project_workload(project, time_horizon) for project in projects]

# Computes the weekly workload matrix W[i,t]: workload assigned to manager i at time t
# x_ij represents fractional assignment from manager i to project j
# Adjusts for manager productivity
def compute_W(x_ij, project_workloads, managers):
    num_managers = len(managers)
    num_projects = len(project_workloads)
    time_horizon = len(project_workloads[0])
    W = np.zeros((num_managers, time_horizon))

    for i in range(num_managers):
        prod = managers[i]['productivity']
        for j in range(num_projects):
            # Distribute workload proportionally to x_ij and adjust by productivity
            W[i] += (x_ij[i][j] / prod) * project_workloads[j]

    return W

# Main objective function to minimize:
# Combines total workload variance (V) and overload penalty (P)
# alpha and beta weight the importance of each term
def objective_function(x_ij, projects, managers, time_horizon, alpha=1.0, beta=1.0):
    project_workloads = compute_all_workloads(projects, time_horizon)
    W = compute_W(x_ij, project_workloads, managers)

    # Deviation from contract hours (even if not overloaded)
    V = np.mean([
        (W[i, t] - managers[i]['contract_hours']) ** 2
        for i in range(len(managers))
        for t in range(time_horizon)
    ])

    # Penalize only overloads above contract hours
    P = np.mean([
        max(0, W[i, t] - managers[i]['contract_hours']) ** 2
        for i in range(len(managers))
        for t in range(time_horizon)
    ])

    return alpha * V + beta * P, V, P

# -------------------------
# GA Core Functions
# -------------------------

def initialize_population(num_managers, num_projects, pop_size):
    return [np.random.dirichlet(np.ones(num_managers), size=num_projects).T.tolist() for _ in range(pop_size)]

def selection(population, fitnesses, num_parents):
    return [x for _, x in sorted(zip(fitnesses, population), key=lambda x: x[0])[:num_parents]]

def crossover(parent1, parent2):
    mask = np.random.rand(*np.array(parent1).shape) < 0.5
    child = np.where(mask, parent1, parent2)
    return normalize_xij(child).tolist()

def mutate(x_ij, mutation_rate=0.01):
    x = np.array(x_ij)
    for j in range(x.shape[1]):
        if np.random.rand() < mutation_rate:
            perturb = np.random.normal(0, 0.1, x.shape[0])
            x[:, j] = np.maximum(0, x[:, j] + perturb)
            x[:, j] /= np.sum(x[:, j])
    return clip_xij(x).tolist()

def normalize_xij(x):
    col_sums = np.sum(x, axis=0)
    col_sums[col_sums == 0] = 1.0
    return x / col_sums

# -------------------------
# GA Runner
# -------------------------

def genetic_algorithm(projects, managers, time_horizon, pop_size=50, generations=100, alpha=1.0, beta=1.0):
    num_managers, num_projects = len(managers), len(projects)
    population = initialize_population(num_managers, num_projects, pop_size)

    for _ in range(generations):
        fitnesses = [objective_function(x, projects, managers, time_horizon, alpha, beta)[0] for x in population]
        parents = selection(population, fitnesses, pop_size // 2)
        next_gen = []
        while len(next_gen) < pop_size:
            child = mutate(crossover(*random.sample(parents, 2)))
            if is_feasible(child):
                next_gen.append(child)
        population = next_gen

    fitnesses = [objective_function(x, projects, managers, time_horizon, alpha, beta)[0] for x in population]
    best_idx = np.argmin(fitnesses)
    best_solution = population[best_idx]
    best_fitness, V, P = objective_function(best_solution, projects, managers, time_horizon, alpha, beta)

    return {"solution": best_solution, "fitness": best_fitness, "deviation": V, "overload": P}

# -------------------------
# Custom Input with Dates
# -------------------------
base_phase_weights = [0.10, 0.40, 0.20, 0.10, 0.15, 0.05]
base_phase_durations = [0.21, 0.33, 0.25, 0.13, None, 0.07]

def weeks_between(start_date, end_date):
    return (end_date - start_date).days // 7

# Define project period
model_start_date = datetime(2025, 7, 1)
model_end_date = datetime(2027, 12, 1)
time_horizon = weeks_between(model_start_date, model_end_date) + 1

projects = [
    {'project': "WC14 Seoul", 'size': 1250, 'start_date': datetime(2025, 9, 1), 'end_date': datetime(2027, 12, 1)},
    {'project': "NVvP Voorjaarscongres 2026", 'size': 1250, 'start_date': datetime(2025, 6, 1), 'end_date': datetime(2026, 6, 1)},
    {'project': "ESTIV 2026", 'size': 525, 'start_date': datetime(2025, 3, 1), 'end_date': datetime(2026, 9, 1)},
    {'project': "BSBF 2026", 'size': 1750, 'start_date': datetime(2024, 10, 1), 'end_date': datetime(2027, 1, 1)},
    {'project': "IEEE WCCI 2026", 'size': 1250, 'start_date': datetime(2024, 10, 1), 'end_date': datetime(2026, 9, 1)},
    {'project': "PROSA 2026", 'size': 180, 'start_date': datetime(2025, 3, 1), 'end_date': datetime(2027, 1, 1)},
    {'project': "Grenzlandcongres", 'size': 250, 'start_date': datetime(2025, 7, 1), 'end_date': datetime(2026, 1, 1)},
    {'project': "3 Countries", 'size': 100, 'start_date': datetime(2025, 7, 1), 'end_date': datetime(2026, 11, 1)}
]

# Calculate durations and convert to weeks
for project in projects:
    project['duration'] = weeks_between(project['start_date'], project['end_date']) + 1
    project['start_week'] = weeks_between(model_start_date, project['start_date'])
    project['end_week'] = project['start_week'] + project['duration'] - 1
    pw, wl = generate_phase_windows_and_workload(project['size'], project['duration'], project['start_week'], base_phase_weights, base_phase_durations)
    project['phase_windows'], project['phase_workload'] = pw, wl

managers = [
    {'manager_name': "Desiree", 'contract_hours': 24, 'productivity': 1.0},
    {'manager_name': "Nadia", 'contract_hours': 24, 'productivity': 1.0},
    {'manager_name': "Stephanie", 'contract_hours': 36, 'productivity': 1.0},
    {'manager_name': "Sanne", 'contract_hours': 24, 'productivity': 1.0}
]

# -------------------------
# Run GA
# -------------------------
result = genetic_algorithm(projects, managers, time_horizon)

print("Best Fitness (Z):", result["fitness"])
print("Deviation (V):", result["deviation"])
print("Overload (P):", result["overload"])
print("Best Allocation (x_ij):")
for row in result["solution"]:
    print(row)

# -------------------------
# Plot Workload vs Capacity with Date Labels
# -------------------------
project_workloads = compute_all_workloads(projects, time_horizon)
num_managers = len(managers)
W = np.zeros((num_managers, time_horizon))
for i in range(num_managers):
    for j in range(len(projects)):
        W[i] += (result["solution"][i][j] / managers[i]['productivity']) * project_workloads[j]

total_workload = np.sum(W, axis=0)
total_capacity = np.sum([m['contract_hours'] for m in managers])

weeks = np.arange(time_horizon)
date_labels = [model_start_date + timedelta(weeks=int(w)) for w in weeks]

fig = plt.figure(figsize=(12, 6))
plt.plot(date_labels, total_workload, label="Workload", color='dodgerblue')
plt.axhline(total_capacity, label="Capacity", color='orange', linestyle='--')
plt.fill_between(date_labels, total_capacity, total_workload, where=(total_workload > total_capacity), color='red', alpha=0.2, label="Overload")
plt.xlabel("Date")
plt.ylabel("Hours")
plt.title("Total Workload vs Capacity")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
output_pdf_path = "output/klinkhamer_workload.pdf"
os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
with PdfPages(output_pdf_path) as pdf:
    pdf.savefig(fig)
plt.close(fig)

print(f"✅ Saved klinkhamer workload to {output_pdf_path}")

# -------------------------
# Plot Manager Workload Heatmap
# -------------------------
# Assuming result, projects, managers, and time_horizon are already defined in the script

# Compute workloads per manager per week
project_workloads = compute_all_workloads(projects, time_horizon)
W = compute_W(result["solution"], project_workloads, managers)  # shape: (num_managers, weeks)

# Round values for annotation readability
W_rounded = np.round(W).astype(int)

# Labels with contract hours
manager_labels = [f"{managers[i]['manager_name']}\n({managers[i]['contract_hours']}h)" for i in range(W.shape[0])]

# Create date labels for each week in the horizon
date_labels = [model_start_date + timedelta(weeks=int(w)) for w in range(time_horizon)]

fig_width = max(16, time_horizon / 2)
fig, ax = plt.subplots(figsize=(fig_width, 6))

sns.heatmap(
    W,
    cmap="YlOrRd",
    cbar=True,
    ax=ax,
    xticklabels=[d.strftime('%Y-%m-%d') for d in date_labels],
    yticklabels=manager_labels,
    annot=W_rounded,
    fmt="d",
    annot_kws={"size": 9}
)
ax.set_title("Manager Weekly Workload Heatmap - GA Result")
ax.set_xlabel("Week Start Date")
ax.set_ylabel("Manager")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

output_pdf_path = "output/manager_workload.pdf"
os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
with PdfPages(output_pdf_path) as pdf:
    pdf.savefig(fig)
plt.close(fig)

print(f"✅ Saved manager workload heatmap to {output_pdf_path}")
