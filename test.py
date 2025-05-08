import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)

cmap = plt.get_cmap('RdYlGn')

norm = plt.Normalize(y.min(), y.max())

line_colors = cmap(norm(y))

plt.scatter(x, y, color=line_colors)
plt.show()
