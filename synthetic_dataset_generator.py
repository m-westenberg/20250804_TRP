### /synthetic_dataset_generator.py

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json

# --- Utility Functions ---

# Generates normalized phase weights for each project phase with added Gaussian noise
# Ensures weights remain positive and sum to 1

def generate_phase_weights(base=[0.10, 0.40, 0.20, 0.10, 0.15, 0.05], sigma=0.02):
    phase_weights = np.array(base)
    noise = np.random.normal(0, sigma, len(base))
    phase_weights = np.maximum(0.01, phase_weights + noise)
    phase_weights = phase_weights / phase_weights.sum()
    return phase_weights

# Rounds a list of numbers so they sum to a specific total while preserving original proportions

def round_to_sum(arr, target_sum):
    raw = np.array(arr)
    floored = np.floor(raw).astype(int)
    remainder = target_sum - floored.sum()
    fractions = raw - floored
    indices = np.argsort(fractions)[::-1]
    for i in range(remainder):
        floored[indices[i]] += 1
    return floored.tolist()

# Splits a project duration into 6 phases with 1 fixed week for the fifth phase
# Remaining duration is distributed based on noisy base proportions

def generate_phase_durations(duration, base=[0.21, 0.33, 0.25, 0.13, 0.07], sigma=0.02):
    variable_duration = duration - 1    # one week fixed for phase 5
    base = np.array(base)
    noise = np.random.normal(0, sigma, len(base))
    noisy_phase_durations = np.maximum(0.01, base + noise)
    scaled_phase_durations = (noisy_phase_durations / noisy_phase_durations.sum()) * variable_duration
    rounded_phase_durations = round_to_sum(scaled_phase_durations, variable_duration)
    phase_durations = []
    for i in range(6):
        if i == 4:
            phase_durations.append(1)
        else: 
            phase_durations.append(rounded_phase_durations.pop(0))
    return phase_durations

# Converts years of experience into productivity levels, with caps and ranges

def experience_to_productivity(exp):
    if exp < 3:
        return round(np.random.uniform(0.40, 0.60), 2)
    if exp < 6:
        return round(np.random.uniform(0.60, 0.80), 2)
    if exp < 10:
        return round(np.random.uniform(0.80, 1.00), 2)
    else:
        return 1.00

# --- Dataset Generation ---

pdf = PdfPages("dataset_report.pdf") # combined PDF to store plots
num_datasets = 200

datasets = []   # container for all datasets

# Loop to generate multiple synthetic datasets
for d in range(num_datasets):
    dataset_id = d + 1
    num_weeks = random.randint(104,260) # project horizon in weeks
    num_projects = int(num_weeks * np.random.uniform(0.10, 0.25))   # projects scale with horizon
    num_managers = random.randint(3,10) # number of available managers
    
    projects = []
    for j in range(num_projects):
        project_id = j + 1
        size = random.choice([200, 400, 800])   # total project hours
        duration = random.randint(52, 104)  # duration in weeks (1-2 years)
        start_week = random.randint(0, num_weeks - duration)
        end_week = start_week + duration
        phase_weights = generate_phase_weights()
        phase_durations = generate_phase_durations(duration)

        # Compute the start week of each phase sequentially
        pointer = start_week
        phase_starts = []
        for dur in phase_durations:
            phase_starts.append(pointer)
            pointer += dur
        
        # (start, end) for each phase window
        phase_windows = [(start, start + dur - 1) for start, dur in zip(phase_starts, phase_durations)]
        # Convert weight into actual workload (hours)
        phase_workload = [float(size * w) for w in phase_weights]

        # Store project
        projects.append({
            'project_id': project_id,
            'size': size,
            'duration': duration,
            'start_week': start_week,
            'end_week': end_week,
            'phase_weights': phase_weights,
            'phase_durations': phase_durations,
            'phase_workload': phase_workload,
            'phase_windows': phase_windows
        })

    managers = []
    for i in range(num_managers):
        manager_id = i + 1
        contract_hours = random.choice([24, 32, 40])    # weekly available hours
        experience = random.randint(0, 14)  # years of experience
        productivity = experience_to_productivity(experience)

        # Store manager
        managers.append({
            'manager_id': manager_id,
            'contract_hours': contract_hours,
            'experience': experience,
            'productivity': productivity
        })  

    # Store dataset
    datasets.append({
        'dataset_id': dataset_id,
        'weeks': num_weeks,
        'num_projects': num_projects,
        'num_project_managers': num_managers,
        'projects': projects,
        'project_managers': managers
    })

    # --- Plotting and statistics ---

    total_workload = np.zeros(num_weeks)
    for project in projects:
        for (start, end), hours in zip(project["phase_windows"], project["phase_workload"]):
            weekly_hours = hours / (end - start + 1)
            for t in range(start, end + 1):
                if t < num_weeks:
                    total_workload[t] += weekly_hours

    total_capacity = np.zeros(num_weeks)
    for manager in managers:
        contract_hours = manager["contract_hours"]
        total_capacity += contract_hours

    overload_weeks = total_workload > total_capacity
    weeks = np.arange(num_weeks)

    # Summary statistics
    avg_proj_size = np.mean([project['size'] for project in projects])
    avg_proj_duration = np.mean([project['duration'] for project in projects])
    total_overload_hours = np.sum(total_workload[overload_weeks] - total_capacity[overload_weeks])
    peak_workload = np.max(total_workload)
    avg_weekly_workload = np.mean(total_workload)
    avg_contract_hours = np.mean([manager['contract_hours'] for manager in managers])

    # Create plot of workload vs. capacity
    fig = plt.figure(figsize=(11.7, 8.3))
    plt.plot(weeks, total_workload, label="Workload", color='dodgerblue')
    plt.plot(weeks, total_capacity, label="Capacity", color='orange', linestyle='--')
    plt.fill_between(weeks, total_capacity, total_workload, where=overload_weeks,
                     color='red', alpha=0.2, label="Overload")

    for project in projects:
        plt.axvline(x=project['start_week'], color='green', linestyle=':', alpha=0.5)
        plt.axvline(x=project['end_week'], color='black', linestyle=':', alpha=0.5)

    # Annotated title for summary information
    plt.title(f"Dataset {dataset_id} - "
              f"Weeks: {num_weeks}, Projects: {num_projects}, PMs: {num_managers},\n"
              f"Avg Size: {avg_proj_size:.1f}, Avg Duration: {avg_proj_duration:.1f}, "
              f"Peak: {peak_workload:.1f}, Avg Load: {avg_weekly_workload:.1f}, "
              f"Overload Hrs: {total_overload_hours:.1f}, Avg Contract: {avg_contract_hours:.1f}")
    plt.xlabel("Week")
    plt.ylabel("Hours")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)

pdf.close()

# --- JSON Serialization ---

# Converts numpy types to native Python types for JSON compatibility

def convert_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return obj.item()
    return obj

filename = f"datasets.json"

# Write generated datasets to JSON
with open(filename, "w") as f:
    json.dump(datasets, f, default=convert_for_json, indent=2)

print(f"\n✅ Saved datasets to: {filename}")
print("✅ Combined PDF report saved to: plots/dataset_report.pdf")