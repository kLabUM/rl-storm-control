# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:14:11 2016

@author: abhiram
"""

import numpy as np
from pond_single_test import pond
import matplotlib.pyplot as plt

## --------------- Function ---------------------------
# Translates height in the pond to the discrete variable
# Input class of the pond
# Returns discrete value for the height index of the matrix wrt to max height 
def state2reality(pond):
    h = pond.height/pond.max_height
    h = np.around(h,decimals=1)
    h = int(np.floor(h*10))
    return h
    
# Reward function
def reward(h,valve_position):
    area = 1
    cd = 1
    qout = np.sqrt(2*9.81*h) * valve_position * area * cd
    if h >= 0.8 and h <=1.2:
        if qout >= 0 and qout <= 5:
            return 100
        else:
            return 0
    else:
        return 0

#Action function
def epsi_greedy(Q,state,i):
    if i > 10000:
        epsi = 0.5/np.exp(i/10000)
    else:
        epsi=0.5
    temp = np.random.rand()
    if temp < epsi:
        action = np.random.randint(0,10)
    else:
        action = np.argmax(Q[state,])
    return action/10.0


Q=np.zeros(shape=(11,11))

p = pond(100.0,2.0)
p.timestep=1
x=[]
for i in range(0,1000):
        state = 0
        p.volume = 0
        p.initial_height = 0
        print p.volume
        while p.overflow == 0:
            action = epsi_greedy(Q,state,i)
            qout = p.qout(action)
            qin = 2
            p.dhdt(qin,qout)      
            r = reward(p.height,action)        
            state_n=state2reality(p)
            action = np.around(action,decimals=1)
            action = int(np.floor(action*10))
            Q[state,action] = Q[state,action] + 2*(r+0.5*np.amax(Q[state_n,]))
            state=state_n
            x.append(p.height)
            plt.plot(x)
            plt.show()

        