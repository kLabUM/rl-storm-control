#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 00:16:31 2017

@author: pluto
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def temp_score(network_outflow, flow_bound):
    score = 0.0
    for i in network_outflow:
        if i > flow_bound:
            score+= (i - flow_bound)*10.0
    return score


rain_duration = ['0005','0010','0015','0030','0060','0120','0180','0360','0720','1080','1440']
return_preiod = ['001', '002', '005', '010', '025', '050','100']

x = []
y = []
z = []
baseline = []
no_control = np.load('outflow_baseline.npy').item()
for i in return_preiod:
    model_outflow = np.load('model5_' + i + 'npy.npy').item()
    x.append(i)
    for j in rain_duration:
        temp_name = 'aa_orifices_v3_scs_' + j + 'min_' + i + 'yr.inp'
        z.append(temp_score(model_outflow[temp_name], 0.10))
        baseline.append(temp_score(no_control[temp_name], 0.10))
Z= np.asarray(z)
baseline = np.asarray(baseline)
Z = Z.reshape(7,11)
baseline = baseline.reshape(7,11)

my_cmap = sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=True)
#plt.figure(1)
#plt.subplot(121)
#ax = sns.heatmap(Z, linewidths=.5, cmap="YlGnBu", vmin=0, vmax=20000)
#ax.set_xticklabels(rain_duration)
#ax.set_yticklabels(return_preiod)
#plt.title('Controlled')
#plt.subplot(122)
#ax = sns.heatmap(baseline, linewidths=.5, cmap="YlGnBu", vmin=0, vmax=20000)
#plt.title('Un-controlled')

#fig = plt.figure(2)my_cmap = sns.light_palette("Navy", as_cmap=True)
fig, ax_lst = plt.subplots(1, 1)
ax_lst.contourf(Z,  cmap=my_cmap)
sns.plt.show()
