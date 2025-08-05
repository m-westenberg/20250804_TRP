### /genetic_algorithm.py

import numpy as np
import random
from objective_function import objective_function, compute_all_workloads, is_feasible, clip_xij

# --- Genetic Algorithm Components ---

# Initializes population of solutions (x_ij matrices)
# Each column sums to 1 (project fully assigned across managers)
def initialize_population(num_managers, num_projects, pop_size):
    population = []
    for _ in range(pop_size):
        x = np.random.dirichlet(np.ones(num_managers), size=num_projects).T
        population.append(x.tolist())
    return population

# Selects top individuals based on fitness (lower is better)
def selection(population, fitnesses, num_parents):
    selected = sorted(zip(fitnesses, population), key=lambda x: x[0])
    return [x for _, x in selected[:num_parents]]

# Combines two parent matrices using uniform crossover with random mask
# Result is normalized to ensure feasibility of assignments
def crossover(parent1, parent2):
    parent1, parent2 = np.array(parent1), np.array(parent2)
    mask = np.random.rand(*parent1.shape) < 0.5
    child = np.where(mask, parent1, parent2)
    child = normalize_xij(child)
    return child.tolist()

# Applies mutation to each project column with a given mutation probability
# Adds Gaussian noise and re-normalizes to maintain valid assignment
# Clipped to [0, 1] range
def mutate(x_ij, mutation_rate=0.01):
    x = np.array(x_ij)
    for j in range(x.shape[1]):
        if np.random.rand() < mutation_rate:
            perturb = np.random.normal(0, 0.1, x.shape[0])
            x[:, j] = np.maximum(0, x[:, j] + perturb)
            x[:, j] /= np.sum(x[:, j])  # re-normalize column
    return clip_xij(x).tolist()

# Normalizes assignment matrix so each project (column) sums to 1
# Used to restore feasibility after crossover or mutation
def normalize_xij(x):
    x = np.array(x)
    col_sums = np.sum(x, axis=0)
    col_sums[col_sums == 0] = 1.0   # avoid divide-by-zero
    return x / col_sums

# --- Main Optimization Loop ---

# Runs the full genetic algorithm:
# 1. Initializes population
# 2. Iteratively evolves population via selection, crossover, mutation
# 3. Selects best solution from final population
def genetic_algorithm(projects, managers, time_horizon, pop_size=50, generations=100, alpha=1.0, beta=1.0):
    num_managers = len(managers)
    num_projects = len(projects)
    project_workloads = compute_all_workloads(projects, time_horizon)

    population = initialize_population(num_managers, num_projects, pop_size)

    for gen in range(generations):
        # Evaluate population
        fitnesses = [
            objective_function(x, projects, managers, time_horizon, alpha, beta)[0]
            for x in population
        ]

        # Select best half as parents
        parents = selection(population, fitnesses, pop_size // 2)
        next_gen = []

        # Create new generation
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            if is_feasible(child):
                next_gen.append(child)

        population = next_gen

    # Final selection of best individual
    fitnesses = [
        objective_function(x, projects, managers, time_horizon, alpha, beta)[0]
        for x in population
    ]
    best_idx = np.argmin(fitnesses)
    best_solution = population[best_idx]
    best_fitness, V, P = objective_function(best_solution, projects, managers, time_horizon, alpha, beta)

    return {
        "solution": best_solution,
        "fitness": best_fitness,
        "deviation": V,
        "overload": P
    }
