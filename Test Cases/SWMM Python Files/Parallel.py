import swmm
import matplotlib.pyplot as plt
import seaborn

inp = 'Parallel 2.inp'
swmm.initialize(inp)

pond1 = []
pond2 = []
outflow = []

while not(swmm.is_over()):
    swmm.run_step()
    pond1.append(swmm.get('S1', swmm.DEPTH, swmm.SI))
    pond2.append(swmm.get('S2', swmm.DEPTH, swmm.SI))
    outflow.append(swmm.get('C3', swmm.FLOW, swmm.SI))

plt.figure(1)
fig = plt.gcf()
fig.suptitle("Uncontrolled Parallel Tanks", fontsize=14)
plt.subplot(1, 3, 1)
plt.plot(pond1, label='S1')
plt.ylabel('Depth of Pond - 1(m)')
plt.xlabel('Time Steps (10 Sec)')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(pond2, label='S2')
plt.ylabel('Depth of Pond - 1(m)')
plt.xlabel('Time Steps (10 Sec)')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(outflow, label='C3')
plt.ylabel('Outflow(cu.m/sec)')
plt.xlabel('Time Steps (10 Sec)')
plt.legend()
plt.show()

