import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def temp_score(network_outflow, flow_bound):
    score = 0.0
    for i in network_outflow:
        if i > flow_bound:
            score+= (i - flow_bound)*1.0
    return score


rain_duration = ['0005','0010','0015','0030','0060','0120','0180',
                 '0360','0720','1080','1440']
return_preiod = ['001', '002', '005', '010', '025', '050','100']

z = np.zeros((7,11))
baseline = np.zeros((7,11))
no_control = np.load('outflow_baseline.npy').item()
q = 1
for k in [3, 4, 5]:
    for i in return_preiod:
        model_outflow = np.load('model' + str(k) + '_' + i + 'npy.npy').item()
        for j in rain_duration:
            temp_name = 'aa_orifices_v3_scs_' + j + 'min_' + i + 'yr.inp'
            z[return_preiod.index(i), rain_duration.index(j)] = (temp_score(model_outflow[temp_name], 0.10))
            baseline[return_preiod.index(i), rain_duration.index(j)] = (temp_score(no_control[temp_name], 0.10))
    plt.subplot(2,2,q)
    ax = sns.heatmap(z, linewidths=.5, vmin=0, vmax=np.max(baseline))
    ax.set_xticklabels(rain_duration)
    ax.set_yticklabels(np.flip(return_preiod, 0))
    plt.title('Model-'+ str(k))
    q+= 1

plt.subplot(224)
ax = sns.heatmap(baseline, linewidths=.5, vmin=0, vmax=np.max(baseline))
ax.set_xticklabels(rain_duration)
ax.set_yticklabels(np.flip(return_preiod, 0))
plt.title('Uncontrolled')
sns.plt.show()
