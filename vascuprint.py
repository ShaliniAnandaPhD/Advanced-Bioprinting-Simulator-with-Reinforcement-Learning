# VascuPrint: Multiscale modeling of vascularization in bioprinted tissues
# Based on the paper "Multiscale modeling of vascularization in bioprinted tissues" (2022)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

### Agent-Based Model ###

# Define Vessel class
class Vessel:
    def __init__(self, pos, radius, flow, parent=None):
        self.pos = pos          # Position (x, y, z)
        self.radius = radius    # Vessel radius
        self.flow = flow        # Blood flow rate
        self.parent = parent    # Parent vessel
        self.children = []      # Child vessels

    def update_flow(self):
        # Update flow based on Murray's law
        if self.parent:
            self.flow = self.parent.flow * (self.radius ** 3) / sum(c.radius ** 3 for c in self.parent.children)
        for child in self.children:
            child.update_flow()

    def grow(self, elongation_rate, timestep):
        # Grow vessel by a given elongation rate
        direction = normalize(np.random.rand(3) - 0.5)
        self.pos += direction * elongation_rate * timestep

    def branch(self, probability, angle, min_radius):
        # Branch vessel with a given probability
        if np.random.rand() < probability and self.radius > min_radius:
            direction = rotate(self.pos - self.parent.pos, angle)
            new_pos = self.pos + normalize(direction) * self.radius
            new_radius = self.radius * 0.9
            new_vessel = Vessel(new_pos, new_radius, self.flow, parent=self)
            self.children.append(new_vessel)

# Initialize vessels
vessels = [Vessel((0, 0, 0), 0.1, 1)]

# Simulate vessel growth and branching
timestep = 0.1
elongation_rate = 0.05
branch_probability = 0.1
branch_angle = np.pi / 4
min_radius = 0.01

for t in range(100):
    for vessel in vessels:
        vessel.grow(elongation_rate, timestep)
        vessel.branch(branch_probability, branch_angle, min_radius)
        vessel.update_flow()
    vessels = [v for v in vessels if v.radius > min_radius]

### Computational Fluid Dynamics Model ###

# Define blood properties
blood_density = 1060  # kg/m^3
blood_viscosity = 0.003  # Pa*s

# Define vessel network geometry
vessel_positions = [v.pos for v in vessels]
vessel_radii = [v.radius for v in vessels]
vessel_connections = [(v.parent.pos, v.pos) for v in vessels if v.parent]

# Generate mesh
mesh_size = 0.01
x, y, z = np.meshgrid(np.arange(0, 1, mesh_size),
                      np.arange(0, 1, mesh_size),
                      np.arange(0, 1, mesh_size))

# Simulate blood flow
def navier_stokes(u, t, rho, mu):
    # Navier-Stokes equations for incompressible fluid
    ux, uy, uz, p = u
    ux_t = -rho * (ux * ux + uy * uy + uz * uz) + mu * (ux + uy + uz) - p
    uy_t = -rho * (ux * ux + uy * uy + uz * uz) + mu * (ux + uy + uz) - p
    uz_t = -rho * (ux * ux + uy * uy + uz * uz) + mu * (ux + uy + uz) - p
    p_t = -(ux + uy + uz)
    return ux_t, uy_t, uz_t, p_t

# Initial conditions
ux0 = np.zeros_like(x)
uy0 = np.zeros_like(y)
uz0 = np.zeros_like(z)
p0 = np.zeros_like(x)

# Boundary conditions
ux0[vessel_positions] = [v.flow for v in vessels]

# Solve Navier-Stokes equations
u0 = (ux0, uy0, uz0, p0)
t = np.linspace(0, 1, 100)
sol = odeint(navier_stokes, u0, t, args=(blood_density, blood_viscosity))
ux, uy, uz, p = sol[-1]

### Visualization ###

# Plot vessel network
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for pos, r in zip(vessel_positions, vessel_radii):
    ax.plot([pos[0]], [pos[1]], [pos[2]], 'ro', markersize=r*10)
for conn in vessel_connections:
    ax.plot([conn[0][0], conn[1][0]],
            [conn[0][1], conn[1][1]],
            [conn[0][2], conn[1][2]], 'b-')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Vessel Network')

# Plot blood flow
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.quiver(x, y, z, ux, uy, uz, length=0.1, color='r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Blood Flow')

plt.show()
