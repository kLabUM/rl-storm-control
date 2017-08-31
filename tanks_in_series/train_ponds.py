import sys
sys.path.append("../epa_swmm")
import swmm
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop


# Reward Function
def reward(overflow1, overflow2, outflow):
    """Reward Function"""
    if outflow <= 0.20:
        reward_out = 10.0*outflow
    else:
        reward_out = -outflow*100.0
    return reward_out - overflow1*10.0 - overflow2*10.0


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
    model.add(Dense(50, input_dim=3))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(101))
    sgd = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


# Ponds Testing
inp = 'tanks_series.inp'


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


window_length = 100000
episode_count = 1000.0
sim_time = 7200.0
time = 0
steps = episode_count*sim_time
epsilon = np.linspace(0.3, 0.0001, steps+10)


#  Book Keeping
rewards1_episodes = []
rewards2_episodes = []
network1_loss = []
network2_loss = []
height1_pond_mean = []
height2_pond_mean = []
outflow_mean = []
episode_counter = 0
flooding_pond1 = []
flooding_pond2 = []

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

    flood_track1 = []
    flood_track2 = []

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
    episode_counter += 1
    while episode_time < sim_time:

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
        overflow1 = swmm.get('S5', swmm.FLOODING, swmm.SI)
        overflow2 = swmm.get('S7', swmm.FLOODING, swmm.SI)
        flood_track1.append(overflow1)
        flood_track2.append(overflow2)
        reward_step = reward(overflow1, overflow2, outflow)

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

        if train and episode_time % 10 == 0:
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

    model1.save_weights('tanks_series.h5')
    model2.save_weights('tanks_series.h5')
    flooding_pond1.append(np.mean(flood_track1))
    flooding_pond2.append(np.mean(flood_track2))
    rewards1_episodes.append(np.mean(reward_tracker_pond1))
    rewards2_episodes.append(np.mean(reward_tracker_pond2))
    height1_pond_mean.append(np.mean(height_pond1_tracker))
    height2_pond_mean.append(np.mean(height_pond2_tracker))
    outflow_mean.append(np.mean(outflow_tracker))
np.save('rewards1', rewards1_episodes)
np.save('rewards2', rewards2_episodes)
np.save('flooding_1', flooding_pond1)
np.save('flooding_2', flooding_pond2)
np.save('outflow_mean', outflow_mean)
np.save('outflow_tracker', outflow_tracker)


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

plt.figure(3)
plt.subplot(2, 1, 1)
plt.plot(flooding_pond1)
plt.title('Flooding-pond1')
plt.subplot(2, 1, 2)
plt.plot(flooding_pond2)
plt.title('Flooding-pond2')

plt.show()
