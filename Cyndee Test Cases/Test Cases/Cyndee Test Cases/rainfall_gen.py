import numpy as np
import scipy as spy
from scipy import signal
import matplotlib.pyplot as plt
import seaborn

window = signal.gaussian(12, std=2)

plt.plot(window*2.09)
plt.show()
