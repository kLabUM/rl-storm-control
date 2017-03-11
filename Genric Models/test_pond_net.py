import swmm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pond_net import pond_tracker
start = datetime.now()
def swmm_track(pond, attributes=["depth", "inflow", "outflow", "flooding"]):
    att_commands = {'depth': swmm.DEPTH,
                    'inflow': swmm.INFLOW,
                    'outflow': swmm.FLOW,
                    'flooding': swmm.FLOODING}
    temp = []
    for i in attributes:
        if i == 'outflow':
            temp.append(swmm.get(pond.orifice_id, att_commands[i], swmm.SI))
        else:
            temp.append(swmm.get(pond.pond_id, att_commands[i], swmm.SI))
    temp = np.asarray(temp)
    return temp

def plot_pond(pond, components_tracking, components_bookkeeping, show=True):
    rows = len(components_tracking) + len(components_bookkeeping)
    fig = plt.figure()
    count = 0
    for i in components_tracking:
        count += 1
        fig.add_subplot(rows,1, count)
        plt.plot(pond.tracker_pond[i].data())
    for i in components_bookkeeping:
        count += 1
        fig.add_subplot(rows,1, count)
        plt.plot(pond.bookkeeping[i].data())
    if show == True:
        plt.show()
    else:
        return fig

def plot_network(Ponds_network, components_tracking, components_bookkeeping, show=True):
    rows = len(components_tracking) + len(components_bookkeeping)
    columns = len(Ponds_network)
    pond_count = 0
    fig = plt.figure()
    for j in Ponds_network:
        count = 1 + pond_count
        for i in components_tracking:
            print count
            fig.add_subplot(rows, columns, count)
            plt.plot(j.tracker_pond[i].data())
            plt.title(i)
            count += columns
        count += columns
        for i in components_bookkeeping:
            print count
            fig.add_subplot(rows, columns, count)
            plt.plot(j.bookkeeping[i].data())
            plt.title(i)
            count += columns
        pond_count += 1

    if show == True:
        plt.show()
    else:
        return fig

inp = 'aa1.inp'
swmm.initialize(inp)
time = 0
test1 = pond_tracker('93-49743', '23', 1, 10)
test2 = pond_tracker('93-50077', '24', 1, 10)
test3 = pond_tracker('93-49921', '25', 1, 10)
while not(swmm.is_over()):
    time += 1
    swmm.run_step()
    obs = swmm_track(test1)
    obs1 = swmm_track(test2)
    obs2 = swmm_track(test3)
    obs = np.append(obs, 1.0)
    obs1 = np.append(obs1, 1.0)
    obs2 = np.append(obs2, 1.0)
    test1.replay_memory_update(time,1,1,1,1)
    test2.replay_memory_update(time,1,1,1,1)
    test3.replay_memory_update(time,1,1,1,1)
    test1.tracker_update(obs)
    test2.tracker_update(obs1)
    test3.tracker_update(obs2)
print test1.tracker_pond['gate_position'].data()
print time
print (test1.replay_memory['states'].data())
print 'Duration :', datetime.now() - start
plot_network([test1, test2, test3],['depth','inflow'],[])
