import swmm
from core_network import stacker
import sys
import seaborn
import matplotlib.pyplot as plt

inp = 'bran.inp'
swmm.initialize(inp)

time = 0
# Nodes as List
NODES_LIS = {'93-49743':'OR39',
             '93-49868':'OR34',
             '93-49919':'OR44',
             '93-49921':'OR45',
             '93-50074':'OR38',
             '93-50076':'OR46',
             '93-50077':'OR48',
             '93-50081':'OR47',
             '93-50225':'OR36',
             '93-90357':'OR43',
             '93-90358':'OR35'}

ponds = {}
outflow_network = []

control_nodes = [i for i in NODES_LIS.keys()]
for i in control_nodes:
    ponds[i] = stacker(1)


rainfall = []
while not swmm.is_over():
    swmm.run_step()
    for i in control_nodes:
        ponds[i].update(swmm.get(i, swmm.DEPTH, swmm.SI))
    time += 1
    outflow_network.append(swmm.get('ZOF1', swmm.INFLOW, swmm.SI))
    rainfall.append(swmm.get('102', swmm.PRECIPITATION, swmm.SI))
print 'Simulation Time: ', time

figure_count = 1
plot_count = 0
iter_count = 0
for i in control_nodes:
    iter_count += 1
    f = plt.figure(figure_count)
    plot_count += 1
    f.add_subplot(1, 11, plot_count)
    plt.plot(ponds[i].data())
    plt.title(i)
    if iter_count % 11 == 0:
        plot_count = 0
        figure_count += 1
plt.figure(figure_count+1)
plt.plot(outflow_network)
plt.figure(figure_count+2)
plt.plot(rainfall)
plt.show()
