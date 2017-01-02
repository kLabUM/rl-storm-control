import swmm
import numpy as np
import matplotlib.pylab as plt
import seaborn

swmm.initialize('Parallel_year.inp')

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
plt.plot(outfall_flows, label='Outfall')
plt.xlabel('Time (HH:MM)')
plt.ylabel('Flow $(m^3 s^{-1})$')
plt.legend()
path_plot = 'Parallel_year.png'
plt.savefig(path_plot)
