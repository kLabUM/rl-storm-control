import sys
sys.path.append("../epa_swmm")
import swmm
import numpy as np
import matplotlib.pyplot as plt
import pdb

inp = "aa_orifices_v3_scs_0360min_025yr.inp"
swmm.initialize(inp)

nodes_list = ["93-50077", "93-50074", "93-49919"]
or_list = ["OR48", "OR38", "OR44"]
j = 0
depth = {}
outflow = {}
inflow = {}
for i in nodes_list:
    depth[i] = np.empty([0])
    outflow[i] = np.empty([0])
    inflow[i] = np.empty([0])
while j < 25000:
    j += 1
    swmm.run_step()
    for i in nodes_list:
        depth[i] = np.append(depth[i], swmm.get(i,
                                                swmm.DEPTH, swmm.SI))
        inflow[i] = np.append(inflow[i], swmm.get(i,
                                                  swmm.INFLOW
                                                  , swmm.SI))
        outflow[i] = np.append(outflow[i], swmm.get(or_list[nodes_list.index(i)]
                                                    , swmm.FLOW,
                                                    swmm.SI))

fig = plt.figure()
for i in nodes_list:
    fig.add_subplot(2,3,nodes_list.index(i)+1)
    plt.plot(depth[i])
    plt.title("depth")
    fig.add_subplot(2,3,nodes_list.index(i)+4)
    plt.plot(inflow[i])
    plt.title("inflow")
    print np.sum(outflow[i])
plt.show()

