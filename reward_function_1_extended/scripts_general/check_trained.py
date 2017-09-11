import numpy as np
import matplotlib.pyplot as plt
import swmm
from pond_net import pond_tracker
from dqn_agent import deep_q_agent
from ger_fun import reward_function, epsi_greedy, swmm_track
from ger_fun import build_network, plot_network, swmm_states
from score_scard import score_board
import seaborn as sns
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
nodes_controlled = {'93-49921' : 'OR45',
                    '93-50077' : 'OR48'}
states_controlled = {'93-49921': ['93-50074', '93-49743', '93-50225', '93-49919'],
                     '93-50077': [i for i in NODES_LIS.keys()]}
controlled_ponds = {}
# Check nodes
nodes_controlled_inflow = {'93-49921' : [],
                           '93-50077' : []}
nodes_controlled_outflow = {'93-49921' : [],
                           '93-50077' : []}
for i in nodes_controlled.keys():
    controlled_ponds[i] = pond_tracker(i,
                                       NODES_LIS[i],
                                       len(states_controlled[i]),
                                       10000)
all_nodes = [i for i in NODES_LIS.keys()]
con_nodes = [i for i in nodes_controlled.keys()]
uco_nodes = list(set(all_nodes)-set(con_nodes))
action_space = np.linspace(0.0, 10.0, 101)
uncontrolled_ponds = {}
for i in uco_nodes:
    uncontrolled_ponds[i] = pond_tracker(i,
                                         NODES_LIS[i],
                                         1, 100)

rain_duration = ['0005','0010','0015','0030','0060','0120','0180','0360','0720','1080','1440']
return_preiod = ['025','100','001', '002', '005', '010', '025', '050']
number_modes = ['4']
re_count = 0
for return_counter in return_preiod:
    for model_i in number_modes:
        files_names = []
        for i in rain_duration:
            temp_name = 'aa_orifices_v3_scs_' + i + 'min_' + return_counter + 'yr.inp'
            files_names.append(temp_name)
        # Initialize Neural Networks
        models_ac = {}
        model_counter_name = 'model' + model_i + '_random4'
        for i in nodes_controlled.keys():
            model = target = build_network(len(states_controlled[i]),
                                        len(action_space),
                                        1, 250, 'relu', 0.0)
            model.load_weights(i+model_counter_name)
            target.set_weights(model.get_weights())
            models_ac[i] = [model, target]
        # Initialize Deep Q agents
        agents_dqn = {}
        for i in nodes_controlled.keys():
            temp = deep_q_agent(models_ac[i][0],
                                models_ac[i][1],
                                len(states_controlled[i]),
                                controlled_ponds[i].replay_memory,
                                epsi_greedy)
            agents_dqn[i] = temp

        out = {}
        episode_counter = 0
        time_sim = 0
        # Simulation Time Steps
        episode_count = len(files_names)
        timesteps = episode_count*14500
        time_limit = 14500
        epsilon_value = np.linspace(0.000, 0.00, timesteps+10)
        performance = {}
        outflow_network = {}
        for i in files_names:
            performance[i] = score_board()
        actions_rec = {}

        # RL Stuff
        name_count = 0
        while time_sim < timesteps:
            inp = files_names[name_count]
            print inp
            episode_counter += 1
            episode_timer = 0
            swmm.initialize(inp)
            done = False
            for i in nodes_controlled.keys():
                controlled_ponds[i].forget_past()
            for i in uncontrolled_ponds.keys():
                uncontrolled_ponds[i].forget_past()
            outflow_track = []
            while episode_timer < time_limit:
                episode_timer += 1
                time_sim += 1
                # Take a look at whats happening
                for i in nodes_controlled.keys():
                    agents_dqn[i].state_vector = swmm_states(states_controlled[i],
                                                            swmm.DEPTH)
                # Take action
                for i in nodes_controlled.keys():
                    action_step = agents_dqn[i].actions_q(epsilon_value[time_sim],
                                                        action_space)
                    agents_dqn[i].action_vector = action_step/100.0
                    swmm.modify_setting(controlled_ponds[i].orifice_id,
                                        agents_dqn[i].action_vector)
                # SWMM step
                swmm.run_step()
                outflow = swmm.get('ZOF1', swmm.INFLOW, swmm.SI)
                outflow_track.append(outflow)
                overflows = swmm_states(all_nodes, swmm.FLOODING)
            out[episode_counter] = outflow_track
            outflow_network[files_names[name_count]] = outflow_track
            name_count += 1
        re_count += 1
        model_sacer = 'model' + model_i + '_' + return_counter + 'npy'
        print 'Saver Model :',model_sacer
        np.save(model_sacer, outflow_network)
