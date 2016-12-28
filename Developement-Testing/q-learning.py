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
    index_discrete = pond_class.max_height/(height_discrete_value[0]-1)
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

q_martix = np.zeros(shape=(101, 12))
test_pond = pond(100.0, 2.0)
test_pond.timestep = 1

volume = []
flow = []
reward = []
re = []
# ----------- Dynamic Plot ----------- #


EPISODES = 200
for i in range(0, EPISODES):
    # Initialize new episode
    state = 0
    test_pond.volume = 0
    test_pond.overflow = 0

    epsi = 0.7
    j = 0   # Iterator to break infinite loop in episode
    volume = []
    reward = []
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
        action = int(np.floor(action*10))
        # 5. Update the Q matrix
        q_martix[state, action] = q_martix[state, action]+0.05*(r + 0.6*np.max(q_martix[state_n, ])-q_martix[state, action])
        # 6. Update the state
        state = state_n
        # 7. Record the changes in the behavior
        #flow.append(qout)
        #volume.append(test_pond.height)
    re.append(np.mean(reward))

plt.figure()
plt.plot(re)

plt.figure()
plt.imshow(q_martix)
plt.axes().set_aspect('auto')
plt.colorbar()
plt.show()

