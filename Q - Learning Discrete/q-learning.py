# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:14:11 2016

@author: abhiram
"""
import sys
sys.path.append('/Users/abhiram/Dropbox/Adaptive-systems/Test Cases Algorithms')
sys.path.append('/Users/abhiram/Dropbox/Adaptive-systems/Reward_Functions')

import numpy as np
from pond_single_test import pond
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from reward import reward12
from scipy import signal
import seaborn


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

q_martix = np.zeros(shape=(101, 11))
test_pond = pond(100.0, 2.0)
test_pond.timestep = 1

volume = []
flow = []
reward = []
re = []
# ----------- Dynamic Plot ----------- #


EPISODES = 2
for i in range(0, EPISODES):
    # Initialize new episode
    state = 0
    test_pond.volume = 0
    inflow = signal.gaussian(100, std=20.0)
    test_pond.overflow = 0
    epsi = 0.7
    j = 0   # Iterator to break infinite loop in episode
    reward = []
    h = []
    while test_pond.overflow == 0:
        # LOOP BREAKER #
        j = j + 1
        if j > 50:
            break
        # ------ INFLOW ----- #
        qin = 2
        # Q - Learning #
        # 1. Chooses action
        action = 0.0#epsi_greedy(q_martix, epsi, state)
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
        q_martix[state, action] = q_martix[state, action]+0.005*(r + 0.6*np.max(q_martix[state_n, ])-q_martix[state, action])
        # 6. Update the state
        state = state_n
        # 7. Record the changes in the behavior
        #flow.append(qout)
        h.append(test_pond.height)
    volume.append(np.mean(h))
    re.append(np.mean(reward))

plt.plot(h)
plt.show()
