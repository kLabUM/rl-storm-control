# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:14:11 2016

@author: abhiram
"""

import numpy as np
from pond_single_test import pond
import matplotlib.pyplot as plt
plt.style.use('ggplot')

## --------------- Function ---------------------------
# Translates height in the pond to the discrete variable
# Input class of the pond
# Returns discrete value for the height index of the matrix wrt to max height

def state2reality(pond_class):
    """Returns state to reality"""
    height = pond_class.height/pond_class.max_height
    height = np.around(height, decimals=1)
    height = int(np.floor(height*10))
    return height

# Reward function
def reward(height_in_pond, valve_position):
    """Reward Function"""
    area = 1
    c_discharge = 1
    discharge = np.sqrt(2 * 9.81 * height_in_pond) * valve_position * area * c_discharge
    if height_in_pond >= 0.8 and height_in_pond <= 0.9 and discharge < 2.0:
        return 1.0
    else:
        return 0.0

#Action function
def epsi_greedy(Q,state,i):
    if i > 500:
        epsi = 0.0
    else:
        epsi = 0.5
    temp = np.random.rand()
    if temp < epsi:
        action = np.random.randint(0,10)
    else:
        action = np.argmax(Q[state,])
    return action/10.0

Q = np.zeros(shape=(11,11))
p = pond(100.0,2.0)
p.timestep = 1
x = []
plt.ion()

for i in range(0,1):
        state = 0
        p.volume = 0
        p.overflow = 0
        p.volume = 0 
        print p.volume
        j=0
        while p.overflow == 0:
            j=j+1
            action = epsi_greedy(Q,state,j)
            qout = p.qout(action)
            qin = 2
            p.dhdt(qin,qout)      
            r = reward(p.height,action)        
            state_n=state2reality(p)
            action = np.around(action,decimals=1)
            action = int(np.floor(action*10))
            Q[state,action] = Q[state,action] + 1.0 *(r + 0.6*np.amax(Q[state_n,]))
            if np.linalg.norm(Q) > 0.0:
                Q = Q/np.linalg.norm(Q)
            else:
                Q=Q
            state=state_n
            x.append(p.height)
        print p.volume



        
