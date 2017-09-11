import numpy as np
import matplotlib.pyplot as plt
import sys

data = np.load(sys.argv[1]).item()

plt.subplot(5,1,1)
plt.plot(data["depth"])
plt.subplot(5,1,2)
plt.plot(data["inflow"])
plt.subplot(5,1,3)
plt.plot(data["outflow"])
plt.subplot(5,1,4)
plt.plot(data["actions"])
plt.subplot(5,1,5)
plt.plot(data["flooding"])

plt.show()
