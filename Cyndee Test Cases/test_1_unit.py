import swmm
import matplotlib.pyplot as plt
import numpy as np


inp = "Parallel_year2.inp"
swmm.initialize(inp)

outfall_flows = []
rainfall = []

while not swmm.is_over():
    swmm.run_step()

    outfall_flows.append(swmm.get('Out1', swmm.INFLOW, swmm.SI))
    rainfall.append(swmm.get('SC1', swmm.PRECIPITATION, swmm.SI))


plt.subplot(2,1,1)
plt.plot(rainfall)
plt.title('Precipitation')
plt.ylabel('Intensity (in)')
plt.subplot(2,1,2)

plt.plot(outfall_flows)
plt.xlabel('Time (HH:MM)')
plt.ylabel('Flow $(m^3 s^{-1})$')
plt.show()
