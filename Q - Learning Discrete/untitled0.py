#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:54:38 2017

@author: abhiram
"""
from collections import deque
import numpy as np

state = deque(maxlen=10)
for i in range(100):
    state.append(i)
