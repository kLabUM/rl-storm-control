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
from reward import reward

style.use('ggplot')

## --------------- Function ---------------------------#


def state2reality(pond_class):
    """Returns state to reality"""
    height = pond_class.height/pond_class.max_height
    height = np.around(height, decimals=1)
    height = int(np.floor(height*10))
    return height


# Action function

def epsi_greedy(matrix, state_system, iterator):
    """Action Choice Based on Greedy Search"""
    if iterator > 6000:
        epsi = 0.0
    else:
        epsi = 0.3
    if np.random.rand() < epsi:
        action_system = np.random.randint(0,10)
    else:
        action_system = np.argmax(matrix[state_system,])
    return action_system/10.0

##--------------- IMPLEMENTATION -----------###

q_martix = np.zeros(shape=(11, 11))
test_pond = pond(1000.0,2.0)
test_pond.timestep = 1

volume = []
flow = []
time = []

## ----------- Dynamic Plot ----------- ##
#plt.ion()


EPISODES = 10
for i in range(0, EPISODES):
    # Initialize new episode
    state = 0
    test_pond.volume = 0
    test_pond.overflow = 0
    test_pond.volume = 0

    j = 0   # Iterator to break infinite loop in episode
    while  j < 10000:#test_pond.overflow == 0:
        # LOOP BREAKER #
        j = j + 1
        if j > 10000:
            break

        # ------ INFLOW ----- #
        qin = 10*np.random.uniform(0,1)

        # Q - Learning #
        # 1. Chooses action
        action = epsi_greedy(q_martix, state, j)
        # 2. Implements Action
        qout = test_pond.qout(action)
        test_pond.dhdt(qin, qout)
        # 3. Receive the reward for the action
        r = reward(test_pond.height, action)
        # 4. Identify the state and action in terms of Q matrix
        state_n = state2reality(test_pond)
        action = np.around(action, decimals=1)
        action = int(np.floor(action*10))
        # 5. Update the Q matrix
        q_martix[state, action] = q_martix[state, action]+1.0*(r+0.6*np.amax(q_martix[state_n,]))
        # 5.1 Normalize the Q matrix based on Norm
        if np.linalg.norm(q_martix) > 0.0:
            q_martix = q_martix/np.linalg.norm(q_martix)
        else:
            q_martix = q_martix
        # 6. Update the state
        state = state_n
        # 7. Record the changes in the behavior

        flow.append(qout)
        volume.append(test_pond.height)
        #print q_martix
        #print test_pond.volume
plt.subplot(121)
plt.plot(flow)
plt.xlabel("Time Steps")
plt.ylabel("Flow-Discharge")
plt.subplot(122)
plt.plot(volume)
plt.xlabel("Time Steps")
plt.ylabel("Height Pond")
plt.show()
