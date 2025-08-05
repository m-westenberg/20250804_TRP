### /simulated_annealing.py

import numpy as np
import random
import math
from objective_function import objective_function, compute_all_workloads, is_feasible, clip_xij

# --- Initialization ---

# Generates a random feasible solution matrix (x_ij) where each column sums to 1
# Represents initial assignment of managers to projects
def generate_initial_solution(num_managers, num_projects):
    return np.random.dirichlet(np.ones(num_managers), size=num_projects).T.tolist()

# --- Neighbor Generation ---

# Slightly perturbs a single x[i,j] value with Gaussian noise
# Normalizes project column and clips values to stay within [0,1]
def generate_neighbor(x_ij):
    x = np.array(x_ij)
    i, j = random.randint(0, x.shape[0] - 1), random.randint(0, x.shape[1] - 1)
    delta = np.random.normal(0, 0.05)
    x[i, j] = np.clip(x[i, j] + delta, 0.0, 1.0)
    x[:, j] /= np.sum(x[:, j])
    return clip_xij(x).tolist()

# --- Simulated Annealing ---

# Optimization via probabilistic search:
# Accepts worse solutions with temperature-dependent probability to escape local minima

def simulated_annealing(projects, managers, time_horizon,
                         initial_temp=1.0, cooling_rate=0.995,
                         min_temp=1e-3, max_iter=1000,
                         alpha=1.0, beta=1.0):

    num_managers = len(managers)
    num_projects = len(projects)
    project_workloads = compute_all_workloads(projects, time_horizon)

    # Generate a feasible starting solution
    current = generate_initial_solution(num_managers, num_projects)
    while not is_feasible(current):
        current = generate_initial_solution(num_managers, num_projects)

    current_score, _, _ = objective_function(current, projects, managers, time_horizon, alpha, beta)
    best = current
    best_score = current_score

    T = initial_temp
    for _ in range(max_iter):
        neighbor = generate_neighbor(current)
        if not is_feasible(neighbor):
            continue

        neighbor_score, _, _ = objective_function(neighbor, projects, managers, time_horizon, alpha, beta)
        delta = neighbor_score - current_score

        # Accept better solutions, or worse with probability exp(-delta / T)
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbor
            current_score = neighbor_score

            if current_score < best_score:
                best = current
                best_score = current_score

        T *= cooling_rate
        if T < min_temp:
            break

    final_score, V, P = objective_function(best, projects, managers, time_horizon, alpha, beta)
    return {
        "solution": best,
        "fitness": final_score,
        "deviation": V,
        "overload": P
    }
