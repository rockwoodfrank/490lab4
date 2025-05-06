import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
rows, cols = 4, 4
L0 = 1.0             # Rest length of springs
k = 100.0            # Spring constant
mass = 1.0           # Mass of each node
g = 9.8              # Gravity
alpha = 0.01         # Relaxation step size
max_iters = 5000
tolerance = 1e-4

# --- Initial positions ---
positions = np.zeros((rows, cols, 2))
for i in range(rows):
    for j in range(cols):
        positions[i, j] = np.array([j * L0, -i * L0])  # grid layout

# --- Fixed nodes (top-left and top-right) ---
fixed = np.zeros((rows, cols), dtype=bool)
fixed[0, 0] = True
fixed[0, -1] = True

# --- Spring connections (horizontal and vertical only for simplicity) ---
spring_pairs = []
for i in range(rows):
    for j in range(cols):
        for di, dj in [(1, 0), (0, 1)]:  # vertical and horizontal neighbors
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                spring_pairs.append(((i, j), (ni, nj)))

# --- Relaxation loop ---
for step in range(max_iters):
    forces = np.zeros_like(positions)

    # Apply gravity
    for i in range(rows):
        for j in range(cols):
            if not fixed[i, j]:
                forces[i, j][1] -= mass * g

    # Compute spring forces
    for (i1, j1), (i2, j2) in spring_pairs:
        p1, p2 = positions[i1, j1], positions[i2, j2]
        delta = p1 - p2
        dist = np.linalg.norm(delta)
        if dist == 0:
            continue
        direction = delta / dist
        force = -k * (dist - L0) * direction
        if not fixed[i1, j1]:
            forces[i1, j1] += force
        if not fixed[i2, j2]:
            forces[i2, j2] -= force

    # Update positions only for non-fixed nodes
    for i, j in zip(*np.where(~fixed)):
        positions[i, j] += alpha * forces[i, j]

    # Check convergence
    max_force = np.max(np.linalg.norm(forces[~fixed], axis=-1))
    if max_force < tolerance:
        print(f"Converged in {step} steps.")
        break
else:
    print("Did not converge.")

# --- Plot the final structure ---
plt.figure(figsize=(6, 6))
for (i1, j1), (i2, j2) in spring_pairs:
    p1 = positions[i1, j1]
    p2 = positions[i2, j2]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=1)

# Plot the masses
for i in range(rows):
    for j in range(cols):
        plt.plot(positions[i, j][0], positions[i, j][1], 'bo')

plt.title("4×4 Spring-Mass Grid (Static Equilibrium)")
plt.axis('equal')
plt.grid(True)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.show()
