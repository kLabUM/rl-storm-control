import numpy as np
import swmm
from pond_net import pond_tracker
from dqn_agent import deep_q_agent
from ger_fun import reward_function, epsi_greedy, swmm_track, build_network, plot_network


def pond1_states():
    hp1 = swmm.get('S5', swmm.DEPTH, swmm.SI)
    hp2 = swmm.get('S7', swmm.DEPTH, swmm.SI)
    ip1 = swmm.get('S5', swmm.INFLOW, swmm.SI)
    return np.array([hp1, hp2, ip1])


def pond2_states():
    hp1 = swmm.get('S5', swmm.DEPTH, swmm.SI)
    hp2 = swmm.get('S7', swmm.DEPTH, swmm.SI)
    ip2 = swmm.get('S7', swmm.INFLOW, swmm.SI)
    return np.array([hp1, hp2, ip2])


# Pond Testing
inp = 's.inp'

pond1 = pond_tracker('S5', 'R2', 3, 10000)
pond2 = pond_tracker('S7', 'C8', 3, 10000)
model_pond1 = target_pond1 = model_pond2 = target_pond2 = build_network(3, 101, 4, 50, 'relu', 0.5)
target_pond1.set_weights(model_pond1.get_weights())
target_pond2.set_weights(model_pond2.get_weights())


epsilon_value = np.linspace(0.5, 0.001, 12000)
action_space = np.linspace(0.0, 10.0, 101)

pond1_a = deep_q_agent(model_pond1,
                       target_pond1,
                       3,
                       pond1.replay_memory,
                       epsi_greedy)

pond2_a = deep_q_agent(model_pond2,
                       target_pond2,
                       3,
                       pond1.replay_memory,
                       epsi_greedy)
timesteps = 0
while timesteps < 10000:
    swmm.initialize(inp)
    done = False
    train = True if timesteps > 5000 else False
    episode_time = 0
    observation_pond1 = pond1_states()
    observation_pond2 = pond2_states()
    pond1.forget_past()
    pond2.forget_past()
    print 'new_simulation'
    while episode_time < 3600:
        episode_time += 1
        timesteps += 1
        states_pond1 = observation_pond1
        states_pond2 = observation_pond2
        actions_pond1 = pond1_a.actions_q(states_pond1, epsilon_value[timesteps])
        actions_pond2 = pond2_a.actions_q(states_pond2, epsilon_value[timesteps])
        swmm.modify_setting('R2', actions_pond1/100.0)
        swmm.modify_setting('C8', actions_pond2/100.0)
        swmm.run_step()

        outflow = swmm.get('C8', swmm.FLOW, swmm.SI)
        overflow1 = swmm.get('S5', swmm.FLOODING, swmm.SI)
        overflow2 = swmm.get('S7', swmm.FLOODING, swmm.SI)

        reward_step = reward_function(overflow1, overflow2, outflow)

        states_new_pond1 = pond1_states()
        states_new_pond2 = pond2_states()

        terminal_pond1 = terminal_pond2 = done

        pond1.replay_memory_update(states_pond1,
                                   states_new_pond1,
                                   reward_step,
                                   actions_pond1,
                                   terminal_pond1)

        pond2.replay_memory_update(states_pond2,
                                   states_new_pond2,
                                   reward_step,
                                   actions_pond2,
                                   terminal_pond2)
        temp1 = np.append(swmm_track(pond1), actions_pond1/100.0)
        temp2 = np.append(swmm_track(pond2), actions_pond2/100.0)
        pond1.tracker_update(np.append(temp1, reward_step))
        pond2.tracker_update(np.append(temp2, reward_step))

        #pond1_a.train_q(timesteps)
        #pond2_a.train_q(timesteps)

        if done:
            break
    pond1.record_mean()
    pond2.record_mean()

plot_network([pond1, pond2],
             ['depth', 'inflow'],
             ['mean_rewards'])

