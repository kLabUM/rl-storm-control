import numpy as np
from pond_net import replay_memory_agent, deep_q_agent, epsi_greedy
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
import itertools
import sys

# Simulation parameters
epi_start = float(sys.argv[1])
epi_end = float(sys.argv[2])
load_model_name = sys.argv[3]
save_model_name = sys.argv[4]


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
        model.add(Activation(activation_function))
        model.add(Dropout(dropout))
    model.add(Dense(output_states))
    model.add(Activation('linear'))
    sgd = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

# Reward Function


def reward_function(outflow, depth, flood):
    # Flow part of the reward
    outflow = [1 if i < 0.1 else -(i-0.1)*10 for i in outflow]
    # Weighting the flow values
    weights = [1, 1, 1]
    # Rewards
    flow_reward = (np.dot(outflow, np.transpose(weights)))
    # Depth rewards
    depth = [-0.5*i if i <= 2.0 else -i**2 + 3 for i in depth]
    weights = [1, 1, 1]
    depth_reward = np.dot(depth, np.transpose(weights))
    # flooding reward
    flood = [-1 if i > 0.0 else 0.0 for i in flood]
    weights = [1, 1, 1]
    flood_reward = np.dot(flood, np.transpose(weights))
    # Sum the total reward
    total_reward = flow_reward + depth_reward + flood_reward
    return total_reward


class gate_positions():
    """
    Maintains the gate position of the central controller
    """

    def __init__(self, nodes_list, closing_factor):
        self.current_gate = np.ones((len(nodes_list)))  # All gates are open
        self.closing_factor = closing_factor  # Float value < 1.0
        self.nodes_list = nodes_list

    def update_gates(self, decisions):
        # Update the previous gate positions
        self.current_gate = self.current_gate + self.closing_factor * decisions
        # Check for -ve gates
        self.current_gate = np.maximum(np.zeros((len(self.nodes_list))), self.current_gate)
        # Check for >1 gates
        self.current_gate = np.minimum(np.ones((len(self.nodes_list))), self.current_gate)


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
gates = gate_positions(con_ponds, 0.10)

# Input States - Heights and previous gate positions
# Heights
temp_current_height = np.zeros(len(con_ponds))
temp_old_height = np.zeros(len(con_ponds)) # Add more states for time
# Append them into a array 
temp_height = np.append(temp_current_height, temp_old_height)
# Gate Positions
temp_gate = gates.current_gate
# Input States
input_states = np.append(temp_height, temp_gate)

# Initialize Action value function
model = build_network(len(input_states), 3**(len(con_ponds)), 2, 50, 'relu', 0.0)
target = build_network(len(input_states), 3**(len(con_ponds)), 2, 50, 'relu', 0.0)
# Initialize Action value function
if load_model_name != "i":
    model.load_weights(load_model_name)
    target.set_weights(model.get_weights())

# Allocate actions
temp_acts = itertools.product(range(3), repeat=len(con_ponds))
temp_acts = list(temp_acts)
action_space = np.asarray([[-1 if j == 0 else 0 if j == 1 else 1 for j in i]
                           for i in temp_acts])
# Replay Memory
replay = replay_memory_agent(len(input_states), 100000)

# Deep Q learning agent
prof_x = deep_q_agent(
    model,
    target,
    len(input_states),
    replay.replay_memory,
    epsi_greedy)

# Simulation Time Steps
episode_count = 198  # Increase the episode count
time_sim = 15000  # Update these values
timesteps = episode_count * time_sim
epsilon_value = np.linspace(epi_start, epi_end, episode_count + 10)
# Mean Reward
rewards_episode_tracker = []
outflow_episode_tracker = []

episode_tracker = 0
t_epsi = 0
action = 0
# Reinforcement Learning
while episode_tracker < episode_count:

    if episode_tracker > 0:
        model.load_weights(save_model_name)

    episode_tracker += 1

    inp = 'aa_orifices_v3_scs_0360min_025yr.inp'

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

        # Input States
        input_states = input_states.reshape(1, 2*len(temp_old_height) + len(temp_gate))

        # Action
        q_values = prof_x.ac_model.predict_on_batch(input_states)

        # Policy
        action = epsi_greedy(len(action_space), q_values,
                             epsilon_value[episode_tracker])

        # Implement Action
        gates.update_gates(action_space[action])
        implement_action(gates.current_gate, con_ponds, NODES_LIS)

        # Run step
        swmm.run_step()

        # Receive the reward
        overflow = [
            swmm.get(pond_name, swmm.FLOODING, swmm.SI)
            for pond_name in con_ponds]
        depth = [
            swmm.get(pond_name, swmm.DEPTH, swmm.SI)
            for pond_name in con_ponds]
        outflow = [swmm.get(NODES_LIS[pond_name], swmm.FLOW, swmm.SI)
                   for pond_name in con_ponds]
        reward = reward_function(outflow, depth, overflow)
        reward_sim.append(reward)

        # Update replay memory

        # Heights
        temp_new_height = np.asarray([swmm.get(i, swmm.DEPTH, swmm.SI) for i in con_ponds])

        # Gate Positions
        temp_new_gate = np.asarray(gates.current_gate)


        # Input States
        input_new_states = np.append(np.append(temp_new_height, temp_old_height), np.asarray(gates.current_gate))

        replay.replay_memory_update(input_states,
                                    input_new_states,
                                    reward,
                                    action,
                                    False)
        input_states = input_new_states
        
        temp_old_height = temp_new_height

        # Train
        if episode_timer % 25 == 0:
            update = False
            if t_epsi % 10000 == 0:
                update = True
            prof_x.train_q(update)

    # Store reward values
    rewards_episode_tracker.append(np.mean(np.asarray(reward_sim)))
    model.save(save_model_name)
np.save(save_model_name + "_rewards", rewards_episode_tracker)
