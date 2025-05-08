import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.animation as animation

# --- Parameters ---
rows, cols = 10,15
L0 = 1.0             # Rest length of springs
k = 200.0            # Spring constant
mass = 1.0           # Mass of each node
g = 9.8              # Gravity
dt = 0.01         # Relaxation step size
tolerance = 1e-4
damping = 1
break_length = 3 * L0      # Create a maximum force before the spring breaks

fig, ax = plt.subplots()
# Setup the color mapping
cmap = plt.get_cmap('magma')
norm = plt.Normalize(0, break_length)

# --- Initial positions ---
positions = np.zeros((rows, cols, 2))
velocities = np.zeros((rows, cols, 2))
linPositions = np.zeros((rows*cols, 2))
for i in range(rows):
    for j in range(cols):
        newArray = np.array([j * L0, -i * L0])  # grid layout
        positions[i, j] = newArray
        linPositions[(i*j)+j] = newArray


ax.set(xlim=[0, rows], ylim=[cols * -1, 0])
# plt.plot(scat)

# --- Fixed nodes (top-left and top-right) ---
fixed = np.zeros((rows, cols), dtype=bool)
fixed[0, 0] = True
fixed[0, -1] = True
# fixed[-1, -1] = True
# fixed[-1, 0] = True


# --- Spring connections ---
spring_pairs = []
spring_lines = []
for i in range(rows):
    for j in range(cols):
        for di, dj in [(1, 0), (0, 1), (1,1), (-1,1)]:  # vertical and horizontal neighbors
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                spring_pairs.append(((i, j), (ni, nj)))
                spring_lines.append(ax.plot([positions[i, j][0], positions[ni, nj][0]], [positions[i, j][1], positions[ni, nj][1]], 'b-'))

pts = ax.scatter([pos[0] for pos in linPositions], [pos[1] for pos in linPositions], c="k", s=20)

# --- Relaxation loop ---
converged = False
broken_forces = []
def update(step):
    global converged, break_length, broken_forces, spring_lines
    forces = np.zeros_like(positions)
    # Apply gravity & damping force
    for i in range(rows):
        for j in range(cols):
            if not fixed[i, j]:
                forces[i, j][1] -= mass * g
                forces[i, j] -= (velocities[i, j]) * damping

    # Compute spring forces
    i = 0
    for (i1, j1), (i2, j2) in spring_pairs:
        p1, p2 = positions[i1, j1], positions[i2, j2]
        
        delta = p1 - p2
        dist = np.linalg.norm(delta)
        color = cmap(norm(dist))
        if dist == 0:
            continue
        direction = delta / dist
        force = -k * (dist - L0) * direction
        if [(i1, j1), (i2, j2)] not in broken_forces:
            spring_lines[i][0].set_data([p1[0], p2[0]], [p1[1], p2[1]])
            spring_lines[i][0].set_color(color)
            if (dist - L0) > break_length:
                print("Broken spring")
                broken_forces.append([(i1, j1), (i2, j2)])
                spring_lines[i][0].set_data([0, 0], [0, 0])
            if not fixed[i1, j1]:
                forces[i1, j1] += force
            if not fixed[i2, j2]:
                forces[i2, j2] -= force
        i += 1

    linPositions = []
    # Plot the fixed positions
    for i, j in zip(*np.where(fixed)):
        linPositions.append(positions[i,j])

    # Update positions only for non-fixed nodes
    for i, j in zip(*np.where(~fixed)):
        velocities[i, j] += dt * (forces[i, j] / mass)
        positions[i, j] += velocities[i, j] * dt
        linPositions.append(positions[i,j])

    # Check convergence
    max_force = np.max(np.linalg.norm(forces[~fixed], axis=-1))
    if max_force < tolerance and converged == False:
        print(f"Converged in {step} steps.")
        converged = True
    pts.set_offsets(linPositions)
    return pts, 

plt.title(f"{rows}×{cols} Spring-Mass Grid (Static Equilibrium)")
plt.axis('equal')
plt.grid(True)
plt.xlabel("X Position")
plt.ylabel("Y Position")

ani = animation.FuncAnimation(fig=fig, func=update, frames=600, interval=60)
writer = animation.FFMpegWriter(fps=30)
ani.save(f"{rows}x{cols} {str(datetime.datetime.now())} animated.mp4", writer=writer)
plt.show()
