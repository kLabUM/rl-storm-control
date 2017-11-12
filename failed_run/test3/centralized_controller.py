import numpy as np
import matplotlib.pylab as plt
import swmm
from pond_net import replay_memory_agent, deep_q_agent, epsi_greedy
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
import itertools
import sys

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


# Reward Function
def reward_function(overflow, outflow):
    """Reward Function Test"""
    reward_out = -200*(outflow-0.05)*(outflow-0.15)
    return reward_out - 100.0*(np.sum(overflow))


class gate_positions():
    def __init__(self, nodes_list, closing_factor):

        self.current_gate = np.ones((len(nodes_list))) # All gates are open
        self.closing_factor = closing_factor # Float value < 1.0

    def update_gates(self, decisions):
        # Update the previous gate positions
        self.current_gate = self.current_gate + self.closing_factor*decisions
        # Check for -ve gates
        self.current_gate = np.maximum(np.zeros((len(nodes_list))),self.current_gate)
        # Check for >1 gates
        self.current_gate = np.minimum(np.ones((len(nodes_list))), self.current_gate)

def implement_action(gate_settings, nodes, NODES_LIS):
    # Set the gate positions
    for i in nodes:
        swmm.modify_setting(NODES_LIS[i], gate_settings[nodes.index(i)])


# Nodes
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
nodes_list = ['93-50077','93-50076']

# Gate Positions
gates = gate_positions(nodes_list, 0.1)

# Input States - Heights and previous gate positions
# Heights
temp_height = np.zeros(len(NODES_LIS.keys()))
# Gate Positions
temp_gate = gates.current_gate
# Input States
input_states = np.append(temp_height, temp_gate)
print len(input_states)

# Initialize Action value function
model = target = build_network(len(input_states), 2**(len(nodes_list)) , 3, 100, 'relu', 0.0)
model.load_weights('prof_x_7_9.h5')
target.set_weights(model.get_weights())


# Allocate actions
temp_acts = itertools.product(range(2), repeat=len(nodes_list))
temp_acts = list(temp_acts)
action_space = np.asarray([[-1 if j == 0 else 1 for j in i] for i in temp_acts])

# Replay Memory
replay = replay_memory_agent(len(input_states), 10000)

# Deep Q learning agent
prof_x = deep_q_agent(model, target, len(input_states), replay.replay_memory, epsi_greedy)

# Simulation Time Steps
episode_count = 1 # Increase the epsiode count
time_sim = 74424.0 # Update these values
timesteps = episode_count* time_sim
epsilon_value = 0.0 #np.linspace(epi_start , epi_end, episode_count+10)

# Multiple Storms
rain_duration = ['0005', '0010', '0030','0060','0120','0180','0360','0720','1080','1440']
return_period = ['001','002','005','010','025','050','100']

# Mean Reward
rewards_episode_tracker = []
outflow_episode_tracker = []

train_data = []
for i in rain_duration:
    for j in return_period:
        temp = 'aa_orifices_v3_scs_' + i + 'min_' + j +'yr.inp'
        train_data.append(temp)

episode_tracker = 0
t_epsi = 0
depth_ponds = {}
# Reinforcement Learning

ponds_list = [i for i in NODES_LIS.keys()]
for i in ponds_list:
    depth_ponds[i] = []

while episode_tracker < episode_count:
    t_epsi += 1
    episode_tracker += 1
    # Pick a random storm
    inp ='aa_orifices_v3_scs_0360min_025yr.inp'
    # Reset episode timer
    episode_timer = 0
    # Initialize swmm
    swmm.initialize(inp)
    # Simulation Tracker
    reward_sim = []
    outflow_sim = []
    action_gates = []
    while episode_timer < time_sim:
        t_epsi += 1
        episode_timer += 1
        # Look at whats happening
        # Heights
        temp_height = np.asarray([swmm.get(i, swmm.DEPTH, swmm.SI) for i in NODES_LIS.keys()])
        # Gate Positions
        temp_gate = np.asarray(gates.current_gate)
        # Input States
        input_states = np.append(temp_height, temp_gate).reshape(1, len(NODES_LIS)+len(nodes_list))

        # Action
        q_values = prof_x.ac_model.predict_on_batch(input_states)
        # Policy
        action = epsi_greedy(len(action_space), q_values, 0.0)#epsilon_value[episode_tracker])
        # Implement Action
        gates.update_gates(action_space[action])
        implement_action(gates.current_gate, nodes_list, NODES_LIS)
        action_gates.append(gates.current_gate)
        for i in ponds_list:
            depth_ponds[i].append(swmm.get(i, swmm.DEPTH, swmm.SI))

        # Run step
        swmm.run_step()

        # Receive the reward
        overflow = np.asarray([swmm.get(i, swmm.FLOODING, swmm.SI) for i in NODES_LIS.keys()])
        outflow = swmm.get('ZOF1', swmm.INFLOW, swmm.SI)
        reward = reward_function(overflow, outflow)
        reward_sim.append(reward)
        outflow_sim.append(outflow)

        # Update replay memory
        # Heights
        temp_new_height = np.asarray([swmm.get(i, swmm.DEPTH, swmm.SI) for i in
                                      NODES_LIS.keys()])
        # Gate Positions
        temp_new_gate = np.asarray(gates.current_gate)
        # Input States
        input_new_states = np.append(temp_new_height, temp_new_gate)

        replay.replay_memory_update(input_states,
                                    input_new_states,
                                    reward,
                                    action,
                                    False)
plt.figure(1)
plt.plot(outflow_sim)


plt.figure(2)
plt.subplot(1,2,1)
plt.plot(action_gates)
plt.title('93-50077')
plt.subplot(1,2,2)
plt.plot(action_gates)
plt.title('93-50076')


plt.figure(3)
fig_num = 1
for i in ponds_list:
    plt.subplot(4,3,fig_num)
    plt.plot(depth_ponds[i])
    plt.title(i)
    fig_num += 1
plt.show()

