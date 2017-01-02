
import swmm
import matplotlib.pyplot as plt
import numpy as np


inp = "Parallel_year2.inp"
swmm.initialize(inp)

y = []
while not swmm.is_over():
    swmm.run_step()
    y.append(swmm.get('SC2', swmm.PRECIPITATION, swmm.SI))


plt.plot(y)
plt.show()
