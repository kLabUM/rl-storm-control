#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


plot_conter = 1
for i in range(2,100):
    plt.figure(plot_conter)
    plt.plot(np.load("outflow_mean/outflow_lastprof_"+str(i)+".h5.npy"),
             label=str(i))
    plt.legend()
    if i%10 == 0:
        plot_conter += 1
    print plot_conter

plt.legend()
plt.show()


