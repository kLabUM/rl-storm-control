import matplotlib.pyplot as plt
import numpy as np

def twinboth(ax):
    # Alternately, we could do `newax = ax._make_twin_axes(frameon=False)`
    newax = ax.figure.add_subplot(ax.get_subplotspec(), frameon=False)
    newax.xaxis.set(label_position='top')
    newax.yaxis.set(label_position='right', offset_position='right')
    newax.yaxis.get_label().set_rotation(-90) # Optional...
    newax.yaxis.tick_right()
    newax.xaxis.tick_top()
    return newax

# Generate random data.
x1 = np.random.randn(50)
y1 = np.linspace(0, 1, 50)
x2 = np.random.randn(20)+15.
y2 = np.linspace(10, 20, 20)

# Plot both curves.
fig, ax1 = plt.subplots()

ax1.set(xlabel='x_1', ylabel='y_1')
ax1.plot(x1, y1, c='r')

ax2 = twinboth(ax1)
ax2.set(xlabel='x_2', ylabel='y_2')
ax2.plot(x2, y2, c='b')

plt.show()
