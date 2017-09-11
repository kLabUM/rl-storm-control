import numpy as np
import matplotlib.pyplot as plt
import swmm
from pond_net import pond_tracker
from ger_fun import plot_network, swmm_track

rain_duration = ['0005','0010','0015','0030','0060','0120','0180','0360','0720','1080','1440']
return_preiod = ['001', '002', '005', '010', '025', '050','100']

files_names = []
for i in rain_duration:
    for j in return_preiod:
        temp_name = 'aa_orifices_v3_scs_' + i + 'min_' + j + 'yr.inp'
        files_names.append(temp_name)

# Nodes as List
NODES_LIS = {'93-49743' : 'OR39',
             '93-49868' : 'OR34',
             '93-49919' : 'OR44',
             '93-49921' : 'OR45',
             '93-50074' : 'OR38',
             '93-50076' : 'OR46',
             '93-50077' : 'OR48',
             '93-50081' : 'OR47',
             '93-50225' : 'OR36',
             '93-90357' : 'OR43',
             '93-90358' : 'OR35'}
all_nodes = [i for i in NODES_LIS.keys()]
time_limit = 14089
figure_counter = 1
outflow = {}
for inp_name in files_names:
    print inp_name
    outflow[inp_name] = []
    all_ponds = {}
    for i in all_nodes:
        all_ponds[i] = pond_tracker(i, NODES_LIS[i],1, 1)
    episode_timer = 0
    outflow_track = []
    swmm.initialize(inp_name)
    while episode_timer < time_limit:
        episode_timer += 1
        # SWMM step
        swmm.run_step()
        # Record Outflows
        outflow_track.append(swmm.get('ZOF1', swmm.INFLOW, swmm.SI))
        # Track all ponds
        for i in all_ponds.keys():
            temp = swmm_track(all_ponds[i], attributes=["depth", "inflow","outflow","flooding"])
            temp = np.append(temp, np.asarray([1.0, 0.0]))
            all_ponds[i].tracker_update(temp)
    outflow[inp_name] = outflow_track
    plt.figure(figure_counter)
    plt.plot(outflow_track)
    plt.title(inp_name)
    plot_network([all_ponds[i] for i in all_ponds.keys()], ['depth','flooding','inflow','outflow'],[],figure_num=figure_counter)
    figure_counter+=1
plt.show()
