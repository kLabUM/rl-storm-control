#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Rainfalls
rain_duration = ['0005', '0010', '0030','0060','0120','0180','0360','0720','1080','1440']
return_period = ['001','002','005','010','025','050','100']

# Matrix of outflows
over = 1
axes = []
for i in rain_duration:
    for j in return_period:
        ax = plt.subplot(10,7, over)
        plt.plot(np.load("actions_"+i+"_"+j+".npy"))
        over = over + 1
        axes.append(ax)

oir = 0
for i in axes[0:7]:
    i.set_title(return_period[oir])
    oir += 1
oir = 0
for i in axes:
    if axes.index(i) % 7 == 0:
        i.set_ylabel(rain_duration[oir], rotation=0)
        oir += 1
plt.show()

