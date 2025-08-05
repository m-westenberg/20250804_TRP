### /objective_function.py

import numpy as np

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
