import swmm
import matplotlib.pyplot as plt
import seaborn


inp = 'Series 3.inp'
swmm.initialize(inp)

height1 = []
height2 = []
qout = []
t = 0
while t < 2880:
    swmm.run_step()
    swmm.modify_setting('R2', 0.0)
    print swmm.get('S5', swmm.FLOODING, swmm.SI)
    height1.append(swmm.get('S5', swmm.DEPTH, swmm.SI))
    height2.append(swmm.get('S7', swmm.DEPTH, swmm.SI))
    qout.append(swmm.get('C8', swmm.FLOW, swmm.SI))
    t = t + 1

plt.figure(1)
fig = plt.gcf()
fig.suptitle("Uncontrolled Series Tanks", fontsize=14)
plt.subplot(1, 3, 1)
plt.plot(height1, label='S5')
plt.ylabel('Depth of Pond - 1(m)')
plt.xlabel('Time Steps (10 Sec)')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(height2, label='S7')
plt.ylabel('Depth of Pond - 1(m)')
plt.xlabel('Time Steps (10 Sec)')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(qout, label='C8')
plt.ylabel('Outflow(cu.m/sec)')
plt.xlabel('Time Steps (10 Sec)')
plt.legend()
plt.show()

