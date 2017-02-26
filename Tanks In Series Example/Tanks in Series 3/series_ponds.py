import swmm
import copy
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import seaborn

# Reward Function
def reward(height1, height2, outflow):
    """Reward Function"""
    if height1 <= 0.0:
        reward1 = 0.0
    else:
        reward1 = -1.0*height1

    if height2 > 0.0:
        reward2 = -1.0*height2
    else:
        reward2 = 0.0

    if outflow <= 0.25:
        reward_out = 10.0*outflow
    else:
        reward_out = -outflow*12.0
    return 1.5*reward1 + 1.5*reward2 + reward_out


# Policy Epsilon Greedy
def epsi_greedy(action_space, q_values, epsilon):
    """Espilon - Greedy"""
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    else:
        return np.argmax(q_values)


# Replay Memory
def snapshots(snaps, observation, length):
    """Makes snap shots and stacks vertically"""
    rows = len(snaps)
    if rows == length:
        snaps = np.vstack((snaps, observation))
        snaps = snaps[1:length+1, ]
    else:
        snaps = np.vstack((snaps, observation))
    return snaps


def randombatch(sample_size, replay_size):
    """Selects a random batch of samples from the
    given matrix"""
    indx = np.linspace(0, replay_size-1, sample_size)
    indx = np.random.choice(indx, sample_size, replace=False)
    indx.tolist()
    indx = list(map(int, indx))
    return indx


def build_network():
    """Neural Nets Action-Value Function"""
    model = Sequential()
    model.add(Dense(10, input_dim=3))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(101))
    model.add(Activation('linear'))
    sgd = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


# Ponds Testing
inp = 's.inp'


# Q-estimator for Pond 1
model1 = build_network()
target_net1 = build_network()
target_net1.set_weights(model1.get_weights())

# Replay Memory Pond 1
states_pond1 = np.zeros((1, 3))
rewards_pond1 = np.zeros((1, 1))
action_replay_pond1 = np.zeros((1, 1))
states_n_pond1 = np.zeros((1, 3))
terminal_pond1 = np.zeros((1, 1))

# Q-estimator for Pond 2
model2 = build_network()
target_net2 = build_network()
target_net2.set_weights(model2.get_weights())

# Replay Memory Pond 2
states_pond2 = np.zeros((1, 3))
rewards_pond2 = np.zeros((1, 1))
action_replay_pond2 = np.zeros((1, 1))
states_n_pond2 = np.zeros((1, 3))
terminal_pond2 = np.zeros((1, 1))

window_length = 50000
episode_count = 500
time = 0
steps = 120000
epsilon = np.linspace(0.1, 0.0, steps+10)


#  Book Keeping
rewards1_episodes = []
rewards2_episodes = []
network1_loss = []
network2_loss = []
height1_pond_mean = []
height2_pond_mean = []
outflow_mean = []
model1.load_weights('pond1_4.h5')
model2.load_weights('pond2_4.h5')

# Simulation
while time < steps:
    print("Time :", time)
    swmm.initialize(inp)
    reward_tracker_pond1 = []
    action_tracker_pond1 = []

    reward_tracker_pond2 = []
    action_tracker_pond2 = []

    height_pond1_tracker = []
    height_pond2_tracker = []

    outflow_tracker = []
    episode_count += 1

    done = False
    train = True if time > 10000 else False
    loss_check_pond1 = 0
    episode_time = 0
    action_space = np.linspace(0.0, 10.0, 101)

    height_pond1 = swmm.get('S5', swmm.DEPTH, swmm.SI)
    height_pond2 = swmm.get('S7', swmm.DEPTH, swmm.SI)

    inflow_pond1 = swmm.get('S5', swmm.INFLOW, swmm.SI)
    inflow_pond2 = swmm.get('S7', swmm.INFLOW, swmm.SI)

    observation_pond1 = np.array([[height_pond1,
                                   height_pond2,
                                   inflow_pond1]])

    observation_pond2 = np.array([[height_pond1,
                                   height_pond2,
                                   inflow_pond2]])

    while episode_time < 2100:

        episode_time += 1
        time += 1

        state_step_pond1 = observation_pond1
        state_step_pond2 = observation_pond2

        states_pond1 = snapshots(states_pond1,
                                 state_step_pond1,
                                 window_length)

        states_pond2 = snapshots(states_pond2,
                                 state_step_pond2,
                                 window_length)

        q_values_pond1 = model1.predict(state_step_pond1)
        q_values_pond2 = model2.predict(state_step_pond2)

        # Book Keeping
        height_pond1_tracker.append(height_pond1)
        height_pond2_tracker.append(height_pond2)

        # Policy
        action_pond1 = epsi_greedy(action_space, q_values_pond1,
                                   epsilon[min(time, steps)])
        action_pond2 = epsi_greedy(action_space, q_values_pond2,
                                   epsilon[min(time, steps)])

        action_tracker_pond1.append(action_pond1)
        action_tracker_pond2.append(action_pond2)

        action_replay_pond1 = snapshots(action_replay_pond1,
                                        action_pond1,
                                        window_length)

        action_replay_pond2 = snapshots(action_replay_pond2,
                                        action_pond2,
                                        window_length)

        swmm.modify_setting('R2', action_pond1/100.0)
        swmm.modify_setting('C8', action_pond2/100.0)

        swmm.run_step()

        temp_terminal = np.zeros((1, 1))
        temp_terminal[0][0] = done

        terminal_pond1 = snapshots(terminal_pond1,
                                   temp_terminal,
                                   window_length)

        terminal_pond2 = snapshots(terminal_pond2,
                                   temp_terminal,
                                   window_length)

        height_pond1 = swmm.get('S5', swmm.DEPTH, swmm.SI)
        height_pond2 = swmm.get('S7', swmm.DEPTH, swmm.SI)
        inflow_pond1 = swmm.get('S5', swmm.INFLOW, swmm.SI)
        inflow_pond2 = swmm.get('S7', swmm.INFLOW, swmm.SI)
        outflow = swmm.get('C8', swmm.FLOW, swmm.SI)

        outflow_tracker.append(outflow)
        reward_step = reward(height_pond1, height_pond2, outflow)

        rewards_pond1 = snapshots(rewards_pond1,
                                  reward_step,
                                  window_length)

        rewards_pond2 = snapshots(rewards_pond2,
                                  reward_step,
                                  window_length)

        reward_tracker_pond1.append(reward_step)
        reward_tracker_pond2.append(reward_step)

        observation_pond1 = np.array([[height_pond1,
                                       height_pond2,
                                       inflow_pond1]])
        observation_pond2 = np.array([[height_pond1,
                                       height_pond2,
                                       inflow_pond2]])

        state_step_n_pond1 = observation_pond1
        state_step_n_pond2 = observation_pond2

        states_n_pond1 = snapshots(states_n_pond1,
                                   state_step_n_pond1,
                                   window_length)
        states_n_pond2 = snapshots(states_n_pond1,
                                   state_step_n_pond2,
                                   window_length)

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
        if done:
            break

    model1.save_weights('pond1_5.h5')
    model2.save_weights('pond2_5.h5')
    rewards1_episodes.append(np.mean(reward_tracker_pond1))
    rewards2_episodes.append(np.mean(reward_tracker_pond2))
    height1_pond_mean.append(np.mean(height_pond1_tracker))
    height2_pond_mean.append(np.mean(height_pond2_tracker))
    outflow_mean.append(np.mean(outflow_tracker))

for layer in model1.layers:
    print layer.get_weights()

plt.figure(1)
plt.subplot(5, 2, 1)
plt.plot(height_pond1_tracker)
plt.title('Height Pond 1')

plt.subplot(5, 2, 2)
plt.plot(height_pond2_tracker)
plt.title('Height Pond 2')

plt.subplot(5, 2, 3)
plt.plot(reward_tracker_pond1)
plt.title('Reward Pond 1')

plt.subplot(5, 2, 4)
plt.plot(reward_tracker_pond2)
plt.title('Rewards Pond 2')

plt.subplot(5, 2, 5)
plt.plot(rewards1_episodes)
plt.title('Rewards Mean Pond 1')

plt.subplot(5, 2, 6)
plt.plot(rewards2_episodes)
plt.title('Rewards Mean Pond 2')

plt.subplot(5, 2, 7)
plt.plot(height1_pond_mean)
plt.title('Height Mean Pond 1')

plt.subplot(5, 2, 8)
plt.plot(height2_pond_mean)
plt.title('Height Mean Pond 2')

plt.subplot(5, 2, 9)
plt.plot(action_tracker_pond1)
plt.title('Action Pond 1')

plt.subplot(5, 2, 10)
plt.plot(action_tracker_pond2)
plt.title('Action Pond 2')

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(outflow_tracker)
plt.title('Outflow')
plt.subplot(2, 1, 2)
plt.plot(outflow_mean)
plt.title('Outflow mean')
plt.plot()
plt.show()
