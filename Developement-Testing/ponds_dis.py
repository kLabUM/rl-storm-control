# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:14:11 2016

@author: abhiram
"""
import sys
sys.path.insert(0,'/Users/abhiram/Desktop/adaptive-systems/Test Cases Algorithms')
sys.path.insert(0,'/Users/abhiram/Desktop/adaptive-systems/Reward_Functions')

import numpy as np
from pond_single_test import pond
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from reward import reward12
style.use('ggplot')

# --------------- Function ---------------------------#

def state2reality(pond_class, Q_matrix):
    """Discrete Ponds"""
    height_discrete_value = Q_matrix.shape
    index_discrete = pond_class.max_height/(height_discrete_value[0]-2)
    height = pond_class.height/index_discrete
    height = int(np.floor(height))
    return height


# Action function
def epsi_greedy(matrix, epsi, state_system):
    """Action Choice Based on Greedy Search"""
    epsilon = epsi
    if np.random.rand() < epsilon:
        action_system = np.random.randint(0, 10)
    else:
        action_system = np.argmax(matrix[state_system, ])
    return action_system/10.0



# #--------------- IMPLEMENTATION -----------###

q_martix = np.zeros(shape=(21, 11))
test_pond = pond(100.0, 2.0, 0.0)
test_pond.timestep = 1

volume = []
flow = []
reward = []
act = []
# ----------- Dynamic Plot ----------- #


EPISODES = 100
for i in range(0, EPISODES):
    # Initialize new episode
    state = 0
    test_pond.volume = 0
    test_pond.overflow = 0
    test_pond.volume = 0
    epsi = 0.7
    j = 0   # Iterator to break infinite loop in episode
    volume = []
    flow = []
    reward = []
    act = []
    while test_pond.overflow == 0:
        # LOOP BREAKER #
        j = j + 1
        if j > 5000:
            break
        # ------ INFLOW ----- #
        qin = 2.0 # *np.random.uniform(0,1)
        # Q - Learning #
        # 1. Chooses action
        action = epsi_greedy(q_martix, epsi, state)
        # 2. Implements Action
        qout = test_pond.qout(action)
        test_pond.dhdt(qin, qout)
        # 3. Receive the reward for the action
        r = reward12(test_pond.height, action)
        reward.append(r)
        # 4. Identify the state and action in terms of Q matrix
        state_n = state2reality(test_pond, q_martix)
        action = np.around(action, decimals=1)
        action = int(np.floor(action*10))
        act.append(action)
        #print (action)
        # 5. Update the Q matrix
        #print (action)
        #print (state)
        q_martix[state, action] = q_martix[state, action]+0.05*(r + 0.5*np.max(q_martix[state_n, ]) - q_martix[state, action])
        # 6. Update the state
        state = state_n
        # 7. Record the changes in the behavior
        flow.append(qout)
        volume.append(test_pond.height)

print (q_martix)

plt.subplot(221)
plt.hist(flow)
plt.xlabel("Time Steps")
plt.ylabel("Flow-Discharge")
plt.subplot(222)
plt.hist(volume)
plt.xlabel("Time Steps")
plt.ylabel("Height Pond")
plt.subplot(223)
plt.hist(reward)
plt.ylabel("Reward")
plt.subplot(224)
plt.hist(act)
plt.ylabel("action")
plt.show()
