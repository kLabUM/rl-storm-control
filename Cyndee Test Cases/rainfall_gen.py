import numpy as np
import scipy as spy
from scipy import signal
import matplotlib.pyplot as plt
import seaborn

storm = signal.gaussian(12, std=2)
rain_interval = np.linspace(0, 12, 12)

plt.plot(rain_interval, storm*2.09)
plt.title('Rain Event 12 Hours - 2 years')
plt.xlabel('Duration(Hours)')
plt.ylabel('Rainfall Intensity (In)')
plt.show()
