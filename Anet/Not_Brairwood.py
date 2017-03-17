import swmm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from datetime import datetime
from copy_py import numpy_list, snapshots, randombatch, epsi_greedy, reward, swmm_network



def build_network(dim):
    """Neural Nets Action-Value Function"""
    model = Sequential()
    model.add(Dense(50, input_dim=dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(101))
    model.add(Activation('linear'))
    sgd = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


# Input SWMM file
inp = 'aa1.inp'
swmm.initialize(inp)

# Nodes as List
N = ['91-51098',
     '93-49743',
     '93-49839',
     '93-49868',
     '93-49869',
     '93-49870',
     '93-49919',
     '93-49921',
     '93-50074',
     '93-50076',
     '93-50077',
     '93-50081',
     '93-50225',
     '93-50227',
     '93-50228',
     '93-50230',
     '93-90357',
     '93-90358',
     'LOHRRD',
     'OAKVALLEY1',
     'WATERSRD1',
     'WATERSRD2',
     'WATERSRD3']


# Q-estimator for Pond 1
model1 = build_network(7)
target_net1 = build_network(7)
target_net1.set_weights(model1.get_weights())

# Q-estimator for Pond 2
model2 = build_network(5)
target_net2 = build_network(5)
target_net2.set_weights(model2.get_weights())

# Q-estimator for Pond 3
model3 = build_network(8)
target_net3 = build_network(8)
target_net3.set_weights(model3.get_weights())

# Q-estimator for Pond 4
model4 = build_network(7)
target_net4 = build_network(7)
target_net4.set_weights(model4.get_weights())

# Q-estimator for Pond 5
model5 = build_network(23)
target_net5 = build_network(23)
target_net5.set_weights(model5.get_weights())

model1.load_weights('no_hope_p1_rogue112.h5')
model2.load_weights('no_hope_p2_rogue112.h5')
model3.load_weights('no_hope_p3_rogue112.h5')
model4.load_weights('no_hope_p4_rogue112.h5')
model5.load_weights('no_hope_p5_rogue112.h5')


# Replay Memory Pond 1
states_pond1 = np.zeros((1, 7))
rewards_pond1 = np.zeros((1, 1))
action_replay_pond1 = np.zeros((1, 1))
states_n_pond1 = np.zeros((1, 7))
terminal_pond1 = np.zeros((1, 1))

# Replay Memory Pond 2
states_pond2 = np.zeros((1, 5))
rewards_pond2 = np.zeros((1, 1))
action_replay_pond2 = np.zeros((1, 1))
states_n_pond2 = np.zeros((1, 5))
terminal_pond2 = np.zeros((1, 1))

# Replay Memory Pond 3
states_pond3 = np.zeros((1, 8))
rewards_pond3 = np.zeros((1, 1))
action_replay_pond3 = np.zeros((1, 1))
states_n_pond3 = np.zeros((1, 8))
terminal_pond3 = np.zeros((1, 1))

# Replay Memory Pond 4
states_pond4 = np.zeros((1, 7))
rewards_pond4 = np.zeros((1, 1))
action_replay_pond4 = np.zeros((1, 1))
states_n_pond4 = np.zeros((1, 7))
terminal_pond4 = np.zeros((1, 1))

# Replay Memory Pond 5
states_pond5 = np.zeros((1, 23))
rewards_pond5 = np.zeros((1, 1))
action_replay_pond5 = np.zeros((1, 1))
states_n_pond5 = np.zeros((1, 23))
terminal_pond5 = np.zeros((1, 1))


# Reinforcement Learning Properties
window_length = 100000
episode_count = 500
time = 0
steps = 5000
epsilon = np.linspace(0.0001, 0.0001, steps+10)

# Book Keeping
rewards_track = numpy_list(23)
heights_track = numpy_list(23)
flood_track = numpy_list(23)
outflow_track = numpy_list(23)

action_space = np.linspace(0.0, 10.0, 101)

while time < steps:
    swmm.initialize(inp)
    rewards_track_episode = numpy_list(1)
    action_track_episode = numpy_list(5)
    height_track_episode = numpy_list(5)
    flood_track_episode = numpy_list(5)
    outflow_tracker = numpy_list(1)
    episode_count += 1
    episode_time = 0

    train = True if time > 10000 else False

    # States Ponds
    # States - Pond 2
    p1 = [N[16], N[2], N[9], N[8], N[10], N[12], N[11]]
    # States - Pond 21
    p2 = [N[22], N[19], N[4], N[13], N[21]]
    # States - Pond 13
    p3 = [N[18], N[17], N[7], N[8], N[10], N[12], N[11], N[13]]
    # States - Pond 8
    p4 = [N[7], N[8], N[10], N[12], N[11], N[9], N[2]]
    # States - Pond 11
    p5 = N

    observation_pond1 = swmm_network(p1, swmm.DEPTH)
    observation_pond2 = swmm_network(p2, swmm.DEPTH)
    observation_pond3 = swmm_network(p3, swmm.DEPTH)
    observation_pond4 = swmm_network(p4, swmm.DEPTH)
    observation_pond5 = swmm_network(p5, swmm.DEPTH)
    startTime = datetime.now()
    while episode_time < 4000:
        episode_time += 1
        time += 1
        done = False
        state_step_p1 = observation_pond1
        state_step_p2 = observation_pond2
        state_step_p3 = observation_pond3
        state_step_p4 = observation_pond4
        state_step_p5 = observation_pond5

        states_pond1 = snapshots(states_pond1, state_step_p1, window_length)
        states_pond2 = snapshots(states_pond2, state_step_p2, window_length)
        states_pond3 = snapshots(states_pond3, state_step_p3, window_length)
        states_pond4 = snapshots(states_pond4, state_step_p4, window_length)
        states_pond5 = snapshots(states_pond5, state_step_p5, window_length)

        state_step_p11 = np.reshape(state_step_p1, (1, 7))
        state_step_p12 = np.reshape(state_step_p2, (1, 5))
        state_step_p13 = np.reshape(state_step_p3, (1, 8))
        state_step_p14 = np.reshape(state_step_p4, (1, 7))
        state_step_p15 = np.reshape(state_step_p5, (1, 23))
        q_values_pond1 = model1.predict(state_step_p11)
        q_values_pond2 = model2.predict(state_step_p12)
        q_values_pond3 = model3.predict(state_step_p13)
        q_values_pond4 = model4.predict(state_step_p14)
        q_values_pond5 = model5.predict(state_step_p15)

        height_track_episode.update(swmm_network([N[2], N[21], N[13],
                                                  N[11], N[8]], swmm.DEPTH))

        action_pond1 = epsi_greedy(action_space, q_values_pond1,
                                   epsilon[min(time, steps)])
        action_pond2 = epsi_greedy(action_space, q_values_pond2,
                                   epsilon[min(time, steps)])
        action_pond3 = epsi_greedy(action_space, q_values_pond3,
                                   epsilon[min(time, steps)])
        action_pond4 = epsi_greedy(action_space, q_values_pond4,
                                   epsilon[min(time, steps)])
        action_pond5 = epsi_greedy(action_space, q_values_pond5,
                                   epsilon[min(time, steps)])

        action_track_episode.update(np.asarray([action_pond1, action_pond2,
                                                action_pond3, action_pond4,
                                                action_pond5]))

        action_replay_pond1 = snapshots(action_replay_pond1,
                                        action_pond1,
                                        window_length)

        action_replay_pond2 = snapshots(action_replay_pond2,
                                        action_pond2,
                                        window_length)

        action_replay_pond3 = snapshots(action_replay_pond3,
                                        action_pond1,
                                   23     window_length)

        action_replay_pond4 = snapshots(action_replay_pond4,
                                        action_pond2,
                                        window_length)

        action_replay_pond5 = snapshots(action_replay_pond5,
                                        action_pond2,
                                        window_length)

        swmm.modify_setting('23', action_pond1/100.0)  # 2  '93-49743'
        swmm.modify_setting('29', action_pond2/100.0)  # 21 'WATERSRD1'
        swmm.modify_setting('27', action_pond3/100.0)  # 13 '93-50225'
        swmm.modify_setting('25', action_pond4/100.0)  # 8  '93-49921'
        swmm.modify_setting('24', action_pond5/100.0)  # 11 '93-50077'

        swmm.run_step()

        temp_terminal = np.zeros((1, 1))
        temp_terminal[0][0] = done

        terminal_pond1 = snapshots(terminal_pond1, temp_terminal,
                                   window_length)
        terminal_pond2 = snapshots(terminal_pond2, temp_terminal,
                                   window_length)
        terminal_pond3 = snapshots(terminal_pond3, temp_terminal,
                                   window_length)
        terminal_pond4 = snapshots(terminal_pond4, temp_terminal,
                                   window_length)
        terminal_pond5 = snapshots(terminal_pond5, temp_terminal,
                                   window_length)

        outflow = swmm.get('13', swmm.FLOW, swmm.SI)

        outflow_tracker.update(outflow)

        overflow = swmm_network([N[2], N[21], N[13], N[11], N[8]],
                                swmm.FLOODING)

        flood_track_episode.update(overflow)

        reward_step = reward(overflow, outflow)

        rewards_track_episode.update(reward_step)

        rewards_pond1 = snapshots(rewards_pond1, reward_step, window_length)
        rewards_pond2 = snapshots(rewards_pond2, reward_step, window_length)
        rewards_pond3 = snapshots(rewards_pond3, reward_step, window_length)
        rewards_pond4 = snapshots(rewards_pond4, reward_step, window_length)
        rewards_pond5 = snapshots(rewards_pond5, reward_step, window_length)

        observation_pond1 = swmm_network(p1, swmm.DEPTH)
        observation_pond2 = swmm_network(p2, swmm.DEPTH)
        observation_pond3 = swmm_network(p3, swmm.DEPTH)
        observation_pond4 = swmm_network(p4, swmm.DEPTH)
        observation_pond5 = swmm_network(p5, swmm.DEPTH)

        state_step_n_pond1 = observation_pond1
        state_step_n_pond2 = observation_pond2
        state_step_n_pond3 = observation_pond3
        state_step_n_pond4 = observation_pond4
        state_step_n_pond5 = observation_pond5

        states_n_pond1 = snapshots(states_pond1, state_step_n_pond1, window_length)
        states_n_pond2 = snapshots(states_pond2, state_step_n_pond2, window_length)
        states_n_pond3 = snapshots(states_pond3, state_step_n_pond3, window_length)
        states_n_pond4 = snapshots(states_pond4, state_step_n_pond4, window_length)
        states_n_pond5 = snapshots(states_pond5, state_step_n_pond5, window_length)

        if train and episode_time % 5 == 0:
            indx = randombatch(32, states_pond1.shape[0])
            indx = list(map(int, indx))
            states_train = states_pond1[indx]
            reward_train = rewards_pond1[indx]
            states_n_train = states_n_pond1[indx]
            action_train = action_replay_pond1[indx]
            terminal_train = terminal_pond1[indx]

            if time % 10000 == 0:
                target_net1.set_weights(model1.get_weights())

            q_values_train_next = target_net1.predict_on_batch(states_n_train)

            target = model1.predict_on_batch(states_train)

            for i in range(32):
                action_idx = int(action_train[i])

                if terminal_train[i]:
                    target[i][action_idx] = reward_train[i]
                else:
                    target[i][action_idx] = reward_train[i] + 0.99 * np.max(
                        q_values_train_next[i])

            model1.fit(states_train,
                       target,
                       batch_size=32,
                       nb_epoch=1,
                       verbose=0)

        if train and episode_time % 5 == 0:
            indx = randombatch(32, states_pond2.shape[0])
            indx = list(map(int, indx))
            states_train = states_pond2[indx]
            reward_train = rewards_pond2[indx]
            states_n_train = states_n_pond2[indx]
            action_train = action_replay_pond2[indx]
            terminal_train = terminal_pond2[indx]

            if time % 10000 == 0:
                target_net2.set_weights(model2.get_weights())

            q_values_train_next = target_net2.predict_on_batch(states_n_train)

            target = model2.predict_on_batch(states_train)

            for i in range(32):
                action_idx = int(action_train[i])

                if terminal_train[i]:
                    target[i][action_idx] = reward_train[i]
                else:
                    target[i][action_idx] = reward_train[i] + 0.99 * np.max(
                        q_values_train_next[i])

            model2.fit(states_train,
                       target,
                       batch_size=32,
                       nb_epoch=1,
                       verbose=0)

        if train and episode_time % 5 == 0:
            indx = randombatch(32, states_pond3.shape[0])
            indx = list(map(int, indx))
            states_train = states_pond3[indx]
            reward_train = rewards_pond3[indx]
            states_n_train = states_n_pond3[indx]
            action_train = action_replay_pond3[indx]
            terminal_train = terminal_pond3[indx]

            if time % 10000 == 0:
                target_net3.set_weights(model3.get_weights())

            q_values_train_next = target_net3.predict_on_batch(states_n_train)

            target = model3.predict_on_batch(states_train)

            for i in range(32):
                action_idx = int(action_train[i])

                if terminal_train[i]:
                    target[i][action_idx] = reward_train[i]
                else:
                    target[i][action_idx] = reward_train[i] + 0.99 * np.max(
                        q_values_train_next[i])

            model3.fit(states_train,
                       target,
                       batch_size=32,
                       nb_epoch=1,
                       verbose=0)

        if train and episode_time % 5 == 0:
            indx = randombatch(32, states_pond4.shape[0])
            indx = list(map(int, indx))
            states_train = states_pond4[indx]
            reward_train = rewards_pond4[indx]
            states_n_train = states_n_pond4[indx]
            action_train = action_replay_pond4[indx]
            terminal_train = terminal_pond4[indx]

            if time % 10000 == 0:
                target_net4.set_weights(model4.get_weights())

            q_values_train_next = target_net4.predict_on_batch(states_n_train)

            target = model4.predict_on_batch(states_train)

            for i in range(32):
                action_idx = int(action_train[i])

                if terminal_train[i]:
                    target[i][action_idx] = reward_train[i]
                else:
                    target[i][action_idx] = reward_train[i] + 0.99 * np.max(
                        q_values_train_next[i])

            model4.fit(states_train,
                       target,
                       batch_size=32,
                       nb_epoch=1,
                       verbose=0)

        if train and episode_time % 5 == 0:
            indx = randombatch(32, states_pond4.shape[0])
            indx = list(map(int, indx))
            states_train = states_pond5[indx]
            reward_train = rewards_pond5[indx]
            states_n_train = states_n_pond5[indx]
            action_train = action_replay_pond5[indx]
            terminal_train = terminal_pond5[indx]

            if time % 10000 == 0:
                target_net5.set_weights(model5.get_weights())

            q_values_train_next = target_net5.predict_on_batch(states_n_train)

            target = model5.predict_on_batch(states_train)

            for i in range(32):
                action_idx = int(action_train[i])

                if terminal_train[i]:
                    target[i][action_idx] = reward_train[i]
                else:
                    target[i][action_idx] = reward_train[i] + 0.99 * np.max(
                        q_values_train_next[i])

            model5.fit(states_train,
                       target,
                       batch_size=32,
                       nb_epoch=1,
                       verbose=0)
            if done:
                break
    # model1.save_weights('no_hope_p1_rogue112.h5')
    # model2.save_weights('no_hope_p2_rogue112.h5')
    # model3.save_weights('no_hope_p3_rogue112.h5')
    # model4.save_weights('no_hope_p4_rogue112.h5')
    # model5.save_weights('no_hope_p5_rogue112.h5')
    print datetime.now() - startTime

plt.figure(1)
plt.plot(rewards_track.data)

plt.figure(2)
for i in range(1, 5):
    plt.subplot(1, 5, i)
    plt.plot(action_track_episode.data[:, i])

plt.show()


