import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


X = []
Y = []
Z = []


def update(t):
    ax.cla()

    x = np.cos(t/10)
    y = np.sin(t/10)
    z = t/10
    print(x)
    X.append(x)
    Y.append(y)
    Z.append(z)

    ax.scatter(x, y, z, s = 100, marker = 'o')
    ax.plot(X, Y, Z)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 10)


fig = plt.figure(dpi=100)
ax = fig.add_subplot(projection='3d')

ani = FuncAnimation(fig = fig, func = update, frames = 100, interval = 100, repeat = False)

plt.show()