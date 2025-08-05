import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages

# -------------------------
# Helper functions
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

def weeks_between(start_date, end_date):
    return (end_date - start_date).days // 7

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

def compute_all_workloads(projects, time_horizon):
    return [compute_project_workload(project, time_horizon) for project in projects]

def compute_W(x_ij, project_workloads, managers):
    num_managers = len(managers)
    num_projects = len(project_workloads)
    time_horizon = len(project_workloads[0])
    W = np.zeros((num_managers, time_horizon))
    for i in range(num_managers):
        prod = managers[i]['productivity']
        for j in range(num_projects):
            W[i] += (x_ij[i][j] / prod) * project_workloads[j]
    return W

def check_allocation_validity(x_ij):
    x = np.array(x_ij)
    project_sums = np.sum(x, axis=0)
    valid = np.allclose(project_sums, 1.0, atol=1e-6)
    if valid:
        print("✅ Allocation valid: Each project sums to 100% across managers.")
    else:
        print("❌ Allocation invalid: Some projects do not sum to 100%.")
        for j, total in enumerate(project_sums):
            print(f"  Project {j+1} sum: {total:.4f}")

# -------------------------
# Manual Project-to-Manager Allocation
# -------------------------
manual_allocation = [
    [0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5],
]

# Check allocation validity
check_allocation_validity(manual_allocation)

# -------------------------
# Define projects and managers
# -------------------------
model_start_date = datetime(2025, 7, 1)
model_end_date = datetime(2027, 12, 1)
time_horizon = (model_end_date - model_start_date).days // 7 + 1

base_phase_weights = [0.10, 0.40, 0.20, 0.10, 0.15, 0.05]
base_phase_durations = [0.21, 0.33, 0.25, 0.13, None, 0.07]

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
# Compute workloads and plot with date labels
# -------------------------
project_workloads = compute_all_workloads(projects, time_horizon)
W = compute_W(manual_allocation, project_workloads, managers)

W_rounded = np.round(W).astype(int)
manager_labels = [f"{m['manager_name']}\n({m['contract_hours']}h)" for m in managers]

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
ax.set_title("Manual Allocation: Manager Weekly Workload Heatmap")
ax.set_xlabel("Week Start Date")
ax.set_ylabel("Manager")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

output_pdf_path = "output/manual_allocation_manager_workload.pdf"
os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
with PdfPages(output_pdf_path) as pdf:
    pdf.savefig(fig)
plt.close(fig)

print(f"✅ Saved manual allocation manager workload heatmap to {output_pdf_path}")
