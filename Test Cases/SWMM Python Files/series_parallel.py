import swmm
import matplotlib.pyplot as plt
import seaborn


inp = 'Series_Parallel 2.inp'
swmm.initialize(inp)

pond1=[]
pond2=[]
pond3=[]
outflow = []
t = 0
while t < 2500 :
    t = t + 1
    swmm.run_step()
    swmm.modify_setting()
    pond1.append(swmm.get('S1', swmm.DEPTH, swmm.SI))
    pond2.append(swmm.get('S2', swmm.DEPTH, swmm.SI))
    pond3.append(swmm.get('S3', swmm.DEPTH, swmm.SI))
    outflow.append(swmm.get('R3', swmm.FLOW, swmm.SI))

plt.figure(1)
fig = plt.gcf()
fig.suptitle("Uncontrolled Series-Parallel Tanks", fontsize=14)
plt.subplot(1, 4, 1)
plt.plot(pond1, label='S1')
plt.ylabel('Depth of Pond - 1(m)')
plt.xlabel('Time Steps (10 Sec)')
plt.legend()
plt.subplot(1, 4, 2)
plt.plot(pond2, label='S2')
plt.ylabel('Depth of Pond - 1(m)')
plt.xlabel('Time Steps (10 Sec)')
plt.legend()
plt.subplot(1, 4, 3)
plt.plot(pond3, label='S3')
plt.ylabel('Depth of Pond - 1(m)')
plt.xlabel('Time Steps (10 Sec)')
plt.legend()
plt.subplot(1, 4, 4)
plt.plot(outflow, label='C3')
plt.ylabel('Outflow(cu.m/sec)')
plt.xlabel('Time Steps (10 Sec)')
plt.legend()
plt.show()
