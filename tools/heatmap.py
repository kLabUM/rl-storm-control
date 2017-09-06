import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def temp_score(network_outflow, flow_bound):
    score = 0.0
    for i in network_outflow:
        if i > flow_bound:
            score+= (i - flow_bound)*10.0
    return score


rain_duration = ['0005','0010','0015','0030','0060','0120','0180',
                 '0360','0720','1080','1440']
return_preiod = ['001', '002', '005', '010', '025', '050','100']

z = []
met = []
for i in return_preiod:
    model_outflow = np.load('model4_' + i + 'npy.npy').item()
    temp = {'return period': i}
    for j in rain_duration:
        temp_name = 'aa_orifices_v3_scs_' + j + 'min_' + i + 'yr.inp'
        met.append(temp_score(model_outflow[temp_name], 0.10))
        temp[j] = (temp_score(model_outflow[temp_name], 0.10))
    z.append(temp)
