from itertools import permutations
from pond_net import pond_tracker
from dqn_agent import deep_q_agent
from ger_fun import swmm_track
from ger_fun import build_network, plot_network, swmm_states
import numpy as np
import swmm

def control_combinations(num_controllers):
    # Generate Action Space
    # Input : Controller Count(int)
    # Output : Possible Control actions(dict)
    temp = '10'
    action_space = {}
    for i in range(0, num_controllers-1):
        temp += '10'
    a = [x for x in permutations(temp, num_controllers)]
    a = list(set(a))
    for i in range(0, len(a)):
        action_space[i] = a[i]
    return action_space


def epsi_greedy(q_values, epsilon):
    action_space_index = np.linspace(0,
                                     len(q_values),
                                     len(q_values)+1,
                                     dtype=int)
    if np.random.rand() <= epsilon:
        return np.random.choice(action_space_index, 1, False)
    else:
        return np.argmax(q_values)


def valve_changer(previous_postion, action):
    if action == 0:
        temp = max(0, previous_postion - 0.1)
    else:
        temp = min(0, previous_postion + 0.1)
    return temp


def action_implement(action_index, action_space, valve_postions, ponds_list):
    temp = action_space[action_index]
    temp_index = 0
    actions_list = []
    for i in ponds_list:
        temp_valve_pos = valve_changer(valve_postions[ponds_list[temp_index]], temp[temp_index])
        valve_postions[ponds_list[temp_index]] = temp_valve_pos
        actions_list.append(temp_valve_pos)
    return actions_list


# Generate Action Space for the Agent
action_space = control_combinations(5)

# Nodes as List
all_nodes = ['91-51098', '93-49743', '93-49839', '93-49868', '93-49869',
             '93-49870', '93-49919', '93-49921', '93-50074', '93-50076',
             '93-50077', '93-50081', '93-50225', '93-50227', '93-50228',
             '93-50230', '93-90357', '93-90358', 'LOHRRD', 'OAKVALLEY1',
             'WATERSRD1', 'WATERSRD2', 'WATERSRD3']

# Controlled nodes
controlled_nodes = {'93-49743': '23', 'WATERSRD1': '29', '93-50225': '27',
                    '93-49921': '25', '93-50077': '24'}

# Pond Tracker for monitoring Uncontrolled Ponds
uncontrolled_ponds = {}
for i in list(set(all_nodes)-set(controlled_nodes)):
    uncontrolled_ponds[i] = pond_tracker(i, 'No', 0, 0)

# Pond Tracker for monitoring Controlled Ponds
controlled_ponds = {}
for i in controlled_nodes.keys():
    controlled_ponds[i] = pond_tracker(i,
                                       controlled_nodes[i],
                                       len(all_nodes),
                                       10000)
# Action value Function
model = target = build_network(len(all_nodes),
                               len(action_space.keys()),
                               4, 50, 'relu', 0.2)
target.set_weights(model.get_weights())

# Initial Valve Positions
valve_position = {}
for i in controlled_nodes.keys():
    valve_position[i] = 0.0

# Simulation Time Steps
episode_count = 100
time_limit = 4970
timesteps = episode_count*time_limit
epsilon_value = np.linspace(0.5, 0.001, timesteps+10)

# Deep Q Agent
agent_x = deep_q_agent(model,
                       target,
                       len(all_nodes),
                       controlled_ponds[0].replay_memory,
                       epsi_greedy)
time_sim = 0
episode_counter = 0
inp = 'aa1.inp'
# Magic Happens !
while time_sim < timesteps:
    episode_counter += 1
    episode_timer = 0
    swmm.initialize(inp)
    done = False
    for i in controlled_ponds.keys():
        controlled_ponds[i].forget_past()
    for i in uncontrolled_ponds.keys():
        uncontrolled_ponds[i].forget_past()
    outflow_track = []
    print 'New Simulation: ', episode_counter
    while episode_timer < time_limit:
        episode_timer += 1
        time_sim += 1
        # Look around
        agent_x.state_vector = swmm_states(all_nodes, swmm.DEPTH)
        # Take action
        

