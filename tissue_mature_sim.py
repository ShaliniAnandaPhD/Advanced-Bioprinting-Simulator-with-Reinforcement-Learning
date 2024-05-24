# TissueMatureSim: Computational modeling of tissue maturation in bioprinted constructs
# Based on the paper "Computational modeling of tissue maturation in bioprinted constructs" (2023)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

### Tissue Growth Model ###

# Define the partial differential equation for tissue growth
def tissue_growth_pde(c, t, D, k):
    # c: cell concentration (cells/cm^3)
    # t: time (days)
    # D: diffusion coefficient (cm^2/day)
    # k: growth rate constant (1/day)
    
    # Implement the PDE using finite differences
    dc_dt = D * np.diff(c, 2) + k * c * (1 - c)
    return dc_dt

# Define the parameters for tissue growth simulation
D = 0.01  # Diffusion coefficient (cm^2/day)
k = 0.2   # Growth rate constant (1/day)
L = 1.0   # Length of the tissue construct (cm)
T = 30    # Simulation time (days)
N = 100   # Number of spatial grid points

# Initialize the cell concentration
c0 = np.zeros(N)
c0[N//2] = 1.0  # Initial cell concentration at the center

# Solve the PDE using the finite difference method
t = np.linspace(0, T, 1000)
dx = L / (N - 1)
x = np.linspace(0, L, N)
c = odeint(tissue_growth_pde, c0, t, args=(D/dx**2, k))

### Cell Differentiation Model ###

# Define the cell types and their properties
cell_types = ['Stem', 'Progenitor', 'Differentiated']
proliferation_rates = [0.5, 0.3, 0.1]  # Proliferation rates for each cell type
differentiation_rates = [0.1, 0.2, 0.0]  # Differentiation rates for each cell type

# Define the cell differentiation model
def cell_differentiation_model(cell_counts, proliferation_rates, differentiation_rates):
    # Update cell counts based on proliferation and differentiation
    new_cell_counts = cell_counts.copy()
    for i in range(len(cell_types) - 1):
        new_cell_counts[i] += proliferation_rates[i] * cell_counts[i]
        new_cell_counts[i] -= differentiation_rates[i] * cell_counts[i]
        new_cell_counts[i+1] += differentiation_rates[i] * cell_counts[i]
    new_cell_counts[-1] += proliferation_rates[-1] * cell_counts[-1]
    return new_cell_counts

# Initialize cell counts
cell_counts = [1000, 0, 0]  # Initial counts for each cell type

# Simulate cell differentiation over time
days = 30
cell_counts_over_time = []
for _ in range(days):
    cell_counts = cell_differentiation_model(cell_counts, proliferation_rates, differentiation_rates)
    cell_counts_over_time.append(cell_counts)
cell_counts_over_time = np.array(cell_counts_over_time)

### Tissue Functionality Model ###

# Define the tissue functionality model
def tissue_functionality_model(cell_counts, functionality_params):
    # Calculate tissue functionality based on cell counts and functionality parameters
    functionality = 0
    for i in range(len(cell_types)):
        functionality += cell_counts[i] * functionality_params[i]
    return functionality

# Define functionality parameters for each cell type
functionality_params = [0.1, 0.5, 1.0]  # Functionality contribution of each cell type

# Calculate tissue functionality over time
tissue_functionality = []
for cell_counts in cell_counts_over_time:
    functionality = tissue_functionality_model(cell_counts, functionality_params)
    tissue_functionality.append(functionality)

### Visualization ###

# Plot tissue growth over time
fig, ax = plt.subplots()
im = ax.imshow(c, cmap='viridis', origin='lower', extent=[0, L, 0, T], aspect='auto')
ax.set_xlabel('Spatial coordinate (cm)')
ax.set_ylabel('Time (days)')
ax.set_title('Tissue Growth')
fig.colorbar(im, ax=ax, label='Cell concentration (cells/cm^3)')

# Plot cell differentiation over time
fig, ax = plt.subplots()
for i in range(len(cell_types)):
    ax.plot(range(days), cell_counts_over_time[:, i], label=cell_types[i])
ax.set_xlabel('Time (days)')
ax.set_ylabel('Cell count')
ax.set_title('Cell Differentiation')
ax.legend()

# Plot tissue functionality over time
fig, ax = plt.subplots()
ax.plot(range(days), tissue_functionality)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Tissue functionality')
ax.set_title('Tissue Functionality')

plt.show()
