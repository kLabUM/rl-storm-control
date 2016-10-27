# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:51:04 2016

@author: abhiram
"""
import numpy as np

epsi = 0.2
for i in range(0,100):
    temp = np.random.rand()
    if temp < epsi:
        action = np.random.randint(0,10)
    else:
        action = 1