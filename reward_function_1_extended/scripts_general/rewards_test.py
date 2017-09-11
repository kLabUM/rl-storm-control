#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn

a = np.empty([])
for i in range(2,8):
    a = np.append(a,np.load("mean_rewardsprof_"+str(i)+".h5.npy"))

plt.figure(1)
plt.plot(a[1:])
plt.ylabel("Rewards")
plt.xlabel("Episodes")

plt.show()
