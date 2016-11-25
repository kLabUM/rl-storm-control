import swmm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
sns.palplot(sns.color_palette("GnBu_d"))

def open2discrete(opening):
    """Discrete Ponds"""
    index_discrete = 0.01
    opening = float(opening) / index_discrete
    opening = int(np.floor(opening))
    return opening


def height2discrete(height):
    """Discrete Ponds"""
    index_discrete = 0.025
    height = float(height) / index_discrete
    height = int(np.floor(height))
    return height


def epsi_greedy(matrix, epsilon, state):
    """Action Value Function"""
    if np.random.rand() < epsilon:
        action = np.random.randint(0, 10)
        action = action / 10.0
    else:
        action = np.argmax(matrix[state, ])
        action = action * 0.01
    return action  # Percent opening


def reward(height, outflow):
    """Reward Function"""
    if height >= 2.90 and height <= 2.950:
        if outflow > 0 and outflow <= 100:
            return 100.0
        else:
            return 0.0
    elif height >= 2.950 and height < 4.00:
        return -10.0  # (height-3.950)*100.0*(1/0.70)
    elif height >= 4.00:
        return -100.0
    elif height < 2.90:
        return (2.90 - height) * (1 / 2.90) * 100.0
    else:
        return 0.0


Q_matrix = np.zeros(shape=(200, 101))

ALPHA = 0.006
GAMMA = 0.6
EPISODES = 2

flow = []
rewq = []
depth = []
rain = []
for i in range(0, EPISODES):
    #SWMM input file
    INP = 'Testcase.inp'
    swmm.initialize(INP)
    # Physical Parameters to NULL --> Update to file
    state = 0  # Initial State
    t = 0
    rew = []
    inflow = []
    out = []
    volume = []
    dep = []
    actionssss = []
    epsi = 0.6
    while t < 4000:
        # 1. Choose a action
        inflow.append(swmm.get('S3', swmm.INFLOW, swmm.SI))
        height = swmm.get('S3', swmm.DEPTH, swmm.SI)
        height = height2discrete(height)
        state = height
        epsi = 0.99 * epsi
        act = epsi_greedy(Q_matrix, epsi, height)
        # 2. Implement Action
        swmm.modify_setting('R1', act)
        swmm.run_step()  # Run SWMM Time step
        # 3. Receive Reward
        height = swmm.get('S3', swmm.DEPTH, swmm.SI)
        outflow = swmm.get('C1', swmm.FLOW, swmm.SI)
        out.append(outflow)
        r = reward(height, outflow)
        rew.append(r)
        # 4. Q-Matrix Update
        state_n = swmm.get('S3', swmm.DEPTH, swmm.SI)
        action = open2discrete(act)
        actionssss.append(action)
        Q_matrix[state, action] = Q_matrix[state, action] + ALPHA * (
            r + GAMMA * np.max(Q_matrix[state_n, ]) - Q_matrix[state, action])
        state = state_n
        volume.append(swmm.get('C1', swmm.FLOW, swmm.SI))
        dep.append(swmm.get('S3', swmm.DEPTH, swmm.SI))
        rain.append(swmm.get('S1', swmm.PRECIPITATION, swmm.SI))
        t = t + 1
    #ERRORS = swmm.finish()
    swmm.close()
    rewq.append(np.mean(rew))
    flow.append(np.mean(volume))
    depth.append(np.mean(dep))
np.savetxt("Q_matrix.txt", Q_matrix)

## ------- No control Case ----------- ##
inflow_s = []
outflow_s = []
height_s = []
for i in range(0, EPISODES):
    #SWMM input file
    INP = 'Testcase.inp'
    swmm.initialize(INP)
    while not swmm.is_over():
        swmm.run_step()
        inflow_s.append(swmm.get('S3', swmm.INFLOW, swmm.SI))
        outflow_s.append(swmm.get('C1', swmm.FLOW, swmm.SI))
        height_s.append(swmm.get('S3', swmm.DEPTH, swmm.SI))
    swmm.close()

plt.subplot(4,1,1)
plt.plot(rain[0:4000], linewidth=2.0, label='Precipitation')
plt.plot(inflow, linewidth=3.0, label='Inflow')
plt.title('Pond Visualization')

plt.legend(loc='upper right', fontsize='small')

plt.subplot(4,1,2)
plt.plot(height_s[0:4000], linewidth=2.0, label='No Control')
plt.plot(dep, linewidth=3.0, label='Control')
plt.ylabel('Depth')
plt.legend(loc='upper right', fontsize='small')

plt.subplot(4,1,3)
plt.plot(outflow_s[0:4000], linewidth=2.0, label='No Control')
plt.plot(volume, linewidth=3.0, label='Control')
plt.ylabel('Outflow')

plt.legend(loc='upper right', fontsize='small')

plt.subplot(4,1,4)
plt.plot(actionssss, linewidth=3.0)
plt.ylabel('% Opening')


plt.show()
