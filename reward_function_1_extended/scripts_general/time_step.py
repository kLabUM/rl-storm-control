import swmm
import matplotlib.pyplot as plt
inp = 'aa1.inp'

swmm.initialize(inp)
time = 0
outflow = []
while not(swmm.is_over()):
    swmm.run_step()
    outflow.append(swmm.get('13', swmm.FLOW, swmm.SI))
    time += 1
print time
plt.plot(outflow)
plt.show()
