import numpy as np
from pond_net import replay_memory_agent, deep_q_agent, epsi_greedy
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
import itertools
import sys
import swmm

# Simulation parameters
load_model_name = sys.argv[1]

def build_network(input_states,
                  output_states,
                  hidden_layers,
                  nuron_count,
                  activation_function,
                  dropout):
    """
    Build and initialize the neural network with a choice for dropout
    """
    model = Sequential()
    model.add(Dense(nuron_count, input_dim=input_states))
    model.add(Activation(activation_function))
    model.add(Dropout(dropout))
    for i_layers in range(0, hidden_layers - 1):
        model.add(Dense(nuron_count))
        model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))      
        model.add(Activation(activation_function))
        model.add(Dropout(dropout))
    model.add(Dense(output_states))
    model.add(Activation('linear'))
    sgd = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

class gate_positions():
    """
    Maintains the gate position of the central controller
    """

    def __init__(self, nodes_list, closing_factor):

        self.current_gate = np.asarray([0.0, 0.0, 0.0])  # All gates are open
        self.closing_factor = closing_factor  # Float value < 1.0
        self.nodes_list = nodes_list

    def update_gates(self, decisions):
        # Update the previous gate positions
        #print(decisions)
        #print(self.closing_factor * decisions)
        self.current_gate = self.current_gate + self.closing_factor * decisions
        # Check for -ve gates
        self.current_gate = np.maximum(np.zeros((len(self.nodes_list))), self.current_gate)
        # Check for >1 gates
        self.current_gate = np.minimum(np.ones((len(self.nodes_list))), self.current_gate)
        #print(self.current_gate)


def implement_action(gate_settings, nodes, NODES_LIS):
    """
    Implements gate actions on all the controlled ponds
    """
    for i in nodes:
        swmm.modify_setting(NODES_LIS[i], gate_settings[nodes.index(i)])


# Nodes
NODES_LIS = {'93-49743': 'OR39',
             '93-49868': 'OR34',
             '93-49919': 'OR44',
             '93-49921': 'OR45',
             '93-50074': 'OR38',
             '93-50076': 'OR46',
             '93-50077': 'OR48',
             '93-50081': 'OR47',
             '93-50225': 'OR36',
             '93-90357': 'OR43',
             '93-90358': 'OR35'}

# list of gates
nodes_list = [i for i in NODES_LIS.keys()]

# controlled ponds
con_ponds = ['93-50077', "93-50076", "93-49921"]
downstream_ponds = ["93-50081"]

# Gate Positions start with open
gates = gate_positions(con_ponds, 0.05)

# Input States - Heights and previous gate positions
# Heights
temp_height = np.zeros(len(con_ponds))
# Gate Positions
temp_gate = gates.current_gate
# Input States
input_states = np.append(temp_height, temp_gate)

# Initialize Action value function
model = target = build_network(
    len(input_states), 3**(len(con_ponds)), 2, 50, 'relu', 0.0)


if load_model_name != "i":
    model.load_weights(load_model_name)
    target.set_weights(model.get_weights())

# Allocate actions
temp_acts = itertools.product(range(3), repeat=len(con_ponds))
temp_acts = list(temp_acts)
action_space = np.asarray([[-1 if j == 0 else 0 if j == 1 else 1 for j in i]
                           for i in temp_acts])
# Replay Memory
replay = replay_memory_agent(len(input_states), 200000)

# Deep Q learning agent
prof_x = deep_q_agent(
    model,
    target,
    len(input_states),
    replay.replay_memory,
    epsi_greedy)

# Simulation Time Steps
episode_count = 1  # Increase the episode count
time_sim = 30000  # Update these values
timesteps = episode_count * time_sim
epsilon_value = np.linspace(0.01, 0.01, episode_count + 10)
# Mean Reward
rewards_episode_tracker = []
outflow_episode_tracker = []

episode_tracker = 0
t_epsi = 0
all_data={}
# Initialize for each pond
for pond_name in NODES_LIS.keys():
    all_data[pond_name] = {}
    all_data[pond_name]["outflow"] = np.empty([0])
    all_data[pond_name]["inflow"] = np.empty([0])
    all_data[pond_name]["depth"] = np.empty([0])
    all_data[pond_name]["flooding"] = np.empty([0])
    all_data[pond_name]["actions"] = np.empty([0])
# Reinforcement Learning
while episode_tracker < episode_count:

    episode_tracker += 1

    inp = './storm_events/aa_orifices_v3_scs_0360min_025yr.inp'

    # Reset episode timer
    episode_timer = 0

    # Initialize swmm
    swmm.initialize(inp)
    # Simulation Tracker
    reward_sim = []
    outflow_sim = []

    print "episode number :", episode_tracker
    print "exploration :", epsilon_value[episode_tracker]

    while episode_timer < time_sim:
        t_epsi += 1
        episode_timer += 1
        # print(episode_timer)
        # Look at whats happening
        # Heights
        temp_height = np.asarray([swmm.get(i, swmm.DEPTH, swmm.SI) for i in con_ponds])

        # Gate Positions
        temp_gate = np.asarray(gates.current_gate)
        if episode_timer%5000==0:
            print(episode_timer, swmm.get())
        # Input States
        input_states = np.append(temp_height, temp_gate).reshape(1, len(temp_height) + len(temp_gate))

        # Action
        q_values = prof_x.ac_model.predict_on_batch(input_states)

        # Policy
        action = epsi_greedy(len(action_space), q_values, epsilon_value[episode_tracker])

        # Implement Action
        gates.update_gates(action_space[action])
        
        implement_action(gates.current_gate, con_ponds, NODES_LIS)

        # Run step
        swmm.run_step()
        # Store the simulation step in to data
        for pond_name in NODES_LIS.keys():
            all_data[pond_name]["outflow"] = np.append(
                all_data[pond_name]["outflow"], swmm.get(
                    NODES_LIS[pond_name], swmm.FLOW, swmm.SI))
            all_data[pond_name]["inflow"] = np.append(
                all_data[pond_name]["inflow"], swmm.get(
                    pond_name, swmm.INFLOW, swmm.SI))
            all_data[pond_name]["depth"] = np.append(
                all_data[pond_name]["depth"], swmm.get(
                    pond_name, swmm.DEPTH, swmm.SI))
            all_data[pond_name]["flooding"] = np.append(
                all_data[pond_name]["flooding"], swmm.get(
                    pond_name, swmm.FLOODING, swmm.SI))
            if pond_name in con_ponds:
                all_data[pond_name]["actions"] = np.append(
                    all_data[pond_name]["actions"],
                    gates.current_gate[con_ponds.index(pond_name)])
            else:
                all_data[pond_name]["actions"] = np.append(
                    all_data[pond_name]["actions"], 1.0)

np.save(load_model_name+"_sim_data_new_72025", all_data)
