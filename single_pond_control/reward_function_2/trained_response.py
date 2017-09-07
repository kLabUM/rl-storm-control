#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import swmm
from pond_net import pond_tracker
from dqn_agent import deep_q_agent
from ger_fun import reward_function, epsi_greedy, swmm_track
from ger_fun import build_network, plot_network, swmm_states
import random
import sys
import os

load_model = "prof_51.h5"
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

nodes_controlled = {'93-50077' : 'OR48'}
states_controlled = {'93-50077': ['93-50077']}
controlled_ponds = {}


for i in nodes_controlled.keys():
    controlled_ponds[i] = pond_tracker(i,
                                       NODES_LIS[i],
                                       len(states_controlled[i]),
                                       1000000)

all_nodes = [i for i in NODES_LIS.keys()]
con_nodes = [i for i in nodes_controlled.keys()]
uco_nodes = list(set(all_nodes)-set(con_nodes))

action_space = np.linspace(0.0, 10.0, 101)
uncontrolled_ponds = {}
for i in uco_nodes:
    uncontrolled_ponds[i] = pond_tracker(i,
                                         NODES_LIS[i],
                                         1, 100)

# Initialize Neural Networks
models_ac = {}
for i in nodes_controlled.keys():
    model = target = build_network(len(states_controlled[i]),
                                   len(action_space),
                                   2, 50, 'relu', 0.0)
    model.load_weights("trained_weights/" + i + load_model)

    target.set_weights(model.get_weights())
    models_ac[i] = [model, target]

rain_duration = ['0005', '0010', '0030','0060','0120','0180','0360','0720','1080','1440']
return_period = ['001','002','005','010','025','050','100']

# Initialize Deep Q agents
agents_dqn = {}
for i in nodes_controlled.keys():
    temp = deep_q_agent(models_ac[i][0],
                        models_ac[i][1],
                        len(states_controlled[i]),
                        controlled_ponds[i].replay_memory,
                        epsi_greedy)
    agents_dqn[i] = temp


# RL Stuff
for i11 in rain_duration:
    for j11 in return_period:
        inp = 'aa_orifices_v3_scs_' + str(i11) + 'min_' + str(j11) +'yr.inp'
        episode_timer = 0
        print inp
        swmm.initialize(inp)
        done = False
        for i in nodes_controlled.keys():
            controlled_ponds[i].forget_past()
        for i in uncontrolled_ponds.keys():
            uncontrolled_ponds[i].forget_past()
        outflow_track = []
        actions_track = []
        while episode_timer < 40000:
            episode_timer += 1
            # Take a look at whats happening
            for i in nodes_controlled.keys():
                agents_dqn[i].state_vector = swmm_states(states_controlled[i],
                                                        swmm.DEPTH)
            # Take action
            for i in nodes_controlled.keys():
                action_step = agents_dqn[i].actions_q(0.0,
                                                    action_space)
                agents_dqn[i].action_vector = action_step/100.0
                actions_track.append(action_step/100.0)
                swmm.modify_setting(controlled_ponds[i].orifice_id,
                                    agents_dqn[i].action_vector)

            # SWMM step
            swmm.run_step()

            # Receive the new rewards
            outflow = swmm.get('ZOF1', swmm.INFLOW, swmm.SI)
            outflow_track.append(outflow)
            overflows = swmm_states(all_nodes, swmm.FLOODING)

            # Observe the new states
            for i in nodes_controlled.keys():
                agents_dqn[i].state_new_vector = swmm_states(states_controlled[i],
                                                            swmm.DEPTH)
            # Update Replay Memory
            for i in nodes_controlled.keys():
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
                temp = swmm_track(uncontrolled_ponds[i], attributes=["depth", "inflow","outflow","flooding"], controlled=True)
                temp = np.append(temp, np.asarray([1.0, 0.0]))
                uncontrolled_ponds[i].tracker_update(temp)


        np.save('51_prof/response_controlled'+i11+"_"+j11+'.npy', controlled_ponds)
        np.save('51_prof/response_uncontrolled'+i11+"_"+j11+'.npy', uncontrolled_ponds)
        np.save('51_prof/outflow_last_'+i11+"_"+j11+".npy", outflow_track)
        np.save('51_prof/actions_'+i11+"_"+j11+".npy", actions_track)



