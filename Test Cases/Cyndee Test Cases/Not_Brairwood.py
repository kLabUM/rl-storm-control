import swmm
import numpy as np
import matplotlib.pyplot as plt
import seaborn


# Input SWMM file
inp = 'aa.inp'
swmm.initialize(inp)

# Nodes as List
N=['91-51098', '93-49743', '93-49839', '93-49868', '93-49869', '93-49870', '93-49919', '93-49921', '93-50074', '93-50076', '93-50077', '93-50081', '93-50225', '93-50227', '93-50228', '93-50230', '93-90357', '93-90358', 'LOHRRD', 'OAKVALLEY1', 'WATERSRD1', 'WATERSRD2', 'WATERSRD3']



# SWMM Simulation

test = []
test1 = []
while not swmm.is_over():

    test.append(swmm.get(N[1], swmm.DEPTH, swmm.SI))
    test1.append(swmm.get(N[22], swmm.DEPTH, swmm.SI))
    swmm.run_step()

plt.subplot(121)
plt.plot(test)
plt.subplot(122)
plt.plot(test1)
plt.show()

