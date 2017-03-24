import numpy as np
import matplotlib.pyplot as plt
import swmm
from pond_net import pond_tracker
from dqn_agent import deep_q_agent
from ger_fun import reward_function, epsi_greedy, swmm_track
from ger_fun import build_network, plot_network, swmm_states

# Nodes as List
all_nodes = ['91-51098', '93-49743', '93-49839', '93-49868', '93-49869',
             '93-49870', '93-49919', '93-49921', '93-50074', '93-50076',
             '93-50077', '93-50081', '93-50225', '93-50227', '93-50228',
             '93-50230', '93-90357', '93-90358', 'LOHRRD', 'OAKVALLEY1',
             'WATERSRD1', 'WATERSRD2', 'WATERSRD3']

controlled_nodes = {'93-49743': '23', 'WATERSRD1': '29', '93-50225': '27',
                    '93-49921': '25', '93-50077': '24'}

states_controlled = {'93-49743': [all_nodes[16], all_nodes[2], all_nodes[9],
                                  all_nodes[8], all_nodes[10], all_nodes[12],
                                  all_nodes[11]],
                     'WATERSRD1': [all_nodes[22], all_nodes[19], all_nodes[4],
                                   all_nodes[13], all_nodes[21]],
                     '93-50225': [all_nodes[18], all_nodes[17], all_nodes[7],
                                  all_nodes[8], all_nodes[10], all_nodes[12],
                                  all_nodes[11], all_nodes[13]],
                     '93-49921': [all_nodes[7], all_nodes[8], all_nodes[10],
                                  all_nodes[12], all_nodes[11], all_nodes[9],
                                  all_nodes[2]],
                     '93-50077': all_nodes}

uncontrolled_ponds = {}
for i in list(set(all_nodes)-set(controlled_nodes)):
    uncontrolled_ponds[i] = pond_tracker(i, 'No', 0, 0)

controlled_ponds = {}
for i in controlled_nodes.keys():
    controlled_ponds[i] = pond_tracker(i,
                                       controlled_nodes[i],
                                       len(states_controlled[i]),
                                       10000)

action_space = np.linspace(0.0, 10.0, 101)

# Initialize Neural Networks
models_ac = {}
for i in controlled_nodes.keys():
    model = target = build_network(len(states_controlled[i]),
                                   len(action_space),
                                   4, 50, 'relu', 0.5)
    model.load_weights(i+'model')
    target.set_weights(model.get_weights())
    models_ac[i] = [model, target]

# Simulation Time Steps
episode_count = 100
time_limit = 4970
timesteps = episode_count*time_limit
epsilon_value = np.linspace(0.5, 0.001, timesteps+10)

# Initialize Deep Q agents
agents_dqn = {}
for i in controlled_nodes.keys():
    temp = deep_q_agent(models_ac[i][0],
                        models_ac[i][1],
                        len(states_controlled[i]),
                        controlled_ponds[i].replay_memory,
                        epsi_greedy)
    agents_dqn[i] = temp

time_sim = 0
episode_counter = 0
inp = 'aa1.inp'
out = {}
# RL Stuff
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
        # Take a look at whats happening
        for i in controlled_ponds.keys():
            agents_dqn[i].state_vector = swmm_states(states_controlled[i],
                                                     swmm.DEPTH)
        # Take action
        for i in controlled_ponds.keys():
            action_step = agents_dqn[i].actions_q(epsilon_value[time_sim],
                                                  action_space)
            agents_dqn[i].action_vector = action_step/100.0
            swmm.modify_setting(controlled_ponds[i].orifice_id,
                                agents_dqn[i].action_vector)
        # SWMM step
        swmm.run_step()
        # Receive the new rewards
        outflow = swmm.get('13', swmm.FLOW, swmm.SI)
        outflow_track.append(outflow)
        overflows = swmm_states(all_nodes, swmm.FLOODING)
        r_temp = reward_function(overflows, outflow)
        for i in controlled_ponds.keys():
            agents_dqn[i].rewards_vector = r_temp
        # Observe the new states
        for i in controlled_ponds.keys():
            agents_dqn[i].state_new_vector = swmm_states(states_controlled[i],
                                                         swmm.DEPTH)
        # Update Replay Memory
        for i in controlled_ponds.keys():
            controlled_ponds[i].replay_memory_update(agents_dqn[i].state_vector,
                                                     agents_dqn[i].state_new_vector,
                                                     agents_dqn[i].rewards_vector,
                                                     agents_dqn[i].action_vector,
                                                     agents_dqn[i].terminal_vector)
        # Track Controlled ponds
        for i in controlled_ponds.keys():
            temp = swmm_track(controlled_ponds[i], attributes=["depth", "inflow","outflow","flooding"], controlled=True)
            temp = np.append(temp, np.asarray([agents_dqn[i].action_vector, agents_dqn[i].rewards_vector]))
            controlled_ponds[i].tracker_update(temp)
        # Track Uncontrolled ponds
        for i in uncontrolled_ponds.keys():
            temp = swmm_track(uncontrolled_ponds[i], attributes=["depth", "inflow","outflow","flooding"])
            temp = np.append(temp, np.asarray([1.0, 0.0]))
            uncontrolled_ponds[i].tracker_update(temp)
        # Train
        for i in controlled_ponds.keys():
            agents_dqn[i].train_q(time_sim)

    out[episode_counter] = outflow_track
    for i in controlled_ponds.keys():
        controlled_ponds[i].record_mean()

all_pond_t = np.append([controlled_ponds[i] for i in controlled_ponds.keys()],
                       [uncontrolled_ponds[i] for i in uncontrolled_ponds.keys()])

for i in models_ac.keys():
    temp = i + 'model1'
    models_ac[i][0].save(temp)
plot_network([controlled_ponds[i] for i in controlled_ponds.keys()], ['depth', 'gate_position', 'flooding'], ['mean_rewards'], figure_num=1)
f = plt.figure(2)
counter = 1
for i in [uncontrolled_ponds[i] for i in uncontrolled_ponds.keys()]:
    f.add_subplot(5, 4, counter)
    plt.plot(i.tracker_pond['depth'].data())
    plt.title(i.pond_id)
    counter += 1
plt.figure(3)
plt.plot(out[episode_counter])
plt.show()



