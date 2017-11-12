import numpy as np
import matplotlib.pyplot as plt
import swmm
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop


# Reward Function
def reward_function(overflow, outflow):
    """Reward Function Test"""
    reward_out = -1.0*(outflow-0.15)*(outflow-0.05)*200
#    if outflow <= 0.10 and outflow >= 0.05:
#        reward_out = 10.0*outflow
#    else:
#i        reward_out = -10.0*(outflow)
    return reward_out - 100.0*(np.sum(overflow))


# Policy Function
def epsi_greedy(action_space, q_values, epsilon):
    """Epsilon Greedy"""
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    else:
        return np.argmax(q_values)


# SWMM network functions
def swmm_track(pond, attributes=["depth", "inflow", "outflow", "flooding"], controlled=False):
    att_commands = {'depth': swmm.DEPTH,
                    'inflow': swmm.INFLOW,
                    'outflow': swmm.FLOW,
                    'flooding': swmm.FLOODING}
    temp = []
    for i in attributes:
        if i == 'outflow':
            if controlled:
                temp.append(swmm.get(pond.orifice_id, att_commands[i], swmm.SI))
            else:
                temp.append(0.0)
        else:
            temp.append(swmm.get(pond.pond_id, att_commands[i], swmm.SI))
    temp = np.asarray(temp)
    return temp


def build_network(input_states,
                  output_states,
                  hidden_layers,
                  nuron_count,
                  activation_function,
                  dropout):

    model = Sequential()
    model.add(Dense(nuron_count, input_dim=input_states))
    model.add(Activation(activation_function))
    model.add(Dropout(dropout))
    for i_layers in range(0, hidden_layers-1):
        model.add(Dense(nuron_count))
        model.add(Activation(activation_function))
        model.add(Dropout(dropout))
    model.add(Dense(output_states))
    model.add(Activation('linear'))
    sgd = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


def plot_network(Ponds_network, components_tracking,
                 components_bookkeeping, figure_num=1, show=True):
    rows = len(components_tracking) + len(components_bookkeeping)
    columns = len(Ponds_network)
    pond_count = 0
    fig = plt.figure(figure_num)
    for j in Ponds_network:
        count = 1 + pond_count
        for i in components_tracking:
            fig.add_subplot(rows, columns, count)
            plt.plot(j.tracker_pond[i].data())
            plt.title(i + j.pond_id)
            count += columns
        for i in components_bookkeeping:
            fig.add_subplot(rows, columns, count)
            plt.plot(j.bookkeeping[i].data())
            plt.title(i+j.pond_id)
            count += columns
        pond_count += 1

    # if show:
    #     plt.show()
    # else:
    #     return fig


# SWMM Network finder
def swmm_states(Network, state):
    temp = []
    for i in Network:
        temp.append(swmm.get(i, state, swmm.SI))
    temp = np.asarray(temp)
    temp = np.reshape(temp, (1, len(temp)))
    return temp
