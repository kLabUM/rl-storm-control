import sys
sys.path.append("../epa_swmm")
import swmm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

inp = "tanks_series.inp"
swmm.initialize(inp)

# Time Steps Counter
time_steps = 0.0

# Water level, Inflow, Outflow, Overflow
inflow = {"1": np.empty([0]), "2": np.empty([0])}
outflow =  {"1": np.empty([0]), "2": np.empty([0])}
overflow =  {"1": np.empty([0]), "2": np.empty([0])}
water_level =  {"1": np.empty([0]), "2": np.empty([0])}

pond_labels = ['S5', 'S7']
outflow_labels = ['R2', 'C8']

while not(swmm.is_over()):
    swmm.run_step()
    time_steps += 1
    count = 0
    for i in inflow.keys():
        inflow[i] = np.append(inflow[i], swmm.get(pond_labels[count], swmm.INFLOW, swmm.SI))
        outflow[i] = np.append(outflow[i], swmm.get(outflow_labels[count],
                                                    swmm.FLOW, swmm.SI))
        overflow[i] = np.append(overflow[i], swmm.get(pond_labels[count],
                                                      swmm.FLOODING, swmm.SI))
        water_level[i] = np.append(water_level[i],swmm.get(pond_labels[count],
                                                           swmm.DEPTH, swmm.SI))
        count += 1

# Make plots prettier !
sns.set(style="white")

print time_steps
fig = plt.figure(1)

plt.subplot(4,2,1)
plt.plot(inflow["1"])
plt.ylabel("Inflow")

plt.subplot(4,2,2)
plt.plot(inflow["2"])

plt.subplot(4,2,3)
plt.plot(water_level["1"])
plt.ylabel("Height")

plt.subplot(4,2,4)
plt.plot(water_level["2"])


plt.subplot(4,2,5)
plt.plot(overflow["1"])
plt.ylabel("overflow")

plt.subplot(4,2,6)
plt.plot(overflow["2"])


plt.subplot(4,2,7)
plt.plot(outflow["1"])
plt.ylabel("outflow")

plt.subplot(4,2,8)
plt.plot(outflow["2"])

fig.suptitle("Uncontrolled Respose")
plt.show()
