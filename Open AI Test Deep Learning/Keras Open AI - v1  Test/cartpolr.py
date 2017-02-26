import copy
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import wrappers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model
from keras.optimizers import Adam


# Policy Epsilon Greedy
def epsi_greedy(action_space, q_values, epsilon):
    """Espilon-Greedy"""
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
    model.add(Dense(30, input_dim=4))
    model.add(Activation('relu'))
    model.add(Dense(30))
    model.add(Activation('relu'))
    model.add(Dense(2))
    sgd = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


# Q-estimator
model = build_network()
target_net = build_network()
model = load_model('my_model7.h5')
target_net.set_weights(model.get_weights())

# Replay Memory
states = np.zeros((1, 4))
rewards = np.zeros((1, 1))
action_replay = np.zeros((1, 1))
states_n = np.zeros((1, 4))
terminal = np.zeros((1, 1))

# Environment Testing
action_space = np.linspace(0, 1, 2, dtype=int)
env = gym.make('CartPole-v1')
env = wrappers.Monitor(env, "/Users/abhiram/Desktop/untitled folder/")

# Reinforcement Learning Parameters and Book Keeping
window_length = 100000
episode_count = 0
time = 0
steps = 170000
epsilon = np.linspace(0.3, 0.1, 150000+10)
rewards_episodes = []
network_loss = []

# Simulation
while time < steps:
    observation = env.reset()
    observation = np.reshape(observation, (1, 4))
    reward_tracker = []
    episode_count += 1
    done = False
    train = True if time > 10000 else False
    loss_check = 0
    episode_time = 0
    while not(done) and episode_time < 20000:
        episode_time += 1
        time += 1
        state_step = observation
        states = snapshots(states, state_step, window_length)
        #  Loss with out y computes the scores from the network
        q_values = model.predict(state_step)

        action = epsi_greedy(action_space, q_values, epsilon[min(time, 120000)])

        action_replay = snapshots(action_replay, action, window_length)

        observation, reward_step, done, info = env.step(action)

        temp_terminal = np.zeros((1, 1))
        temp_terminal[0][0] = done
        terminal = snapshots(terminal, temp_terminal, window_length)

        rewards = snapshots(rewards, reward_step, window_length)
        reward_tracker.append(reward_step)

        observation = np.reshape(observation, (1, 4))
        state_step_n = observation
        states_n = snapshots(states_n, state_step_n, window_length)

        if False:
            print('How ?')
            indx = randombatch(32, states.shape[0])
            indx = list(map(int, indx))
            states_train = states[indx]
            reward_train = rewards[indx]
            states_n_train = states_n[indx]
            action_train = action_replay[indx]
            terminal_train = terminal[indx]

            if time % 10000 == 0:
                target_net.set_weights(model.get_weights())

            q_values_train_next = target_net.predict_on_batch(states_n_train)

            target = model.predict_on_batch(states_train)

            for i in range(32):
                action_idx = int(action_train[i])

                if terminal_train[i]:
                    target[i][action_idx] = reward_train[i]
                else:
                    target[i][action_idx] = reward_train[i] + 0.99 * np.max(q_values_train_next[i])

            model.fit(states_train,
                      target,
                      batch_size=32,
                      nb_epoch=1,
                      verbose=0)

    rewards_episodes.append(np.sum(reward_tracker))

env.close()
gym.upload("/Users/abhiram/Desktop/untitled folder/", api_key='sk_tHy7ubR2T7mu1KmejCddyA')
model.save('my_model8.h5')
plt.figure(1)
plt.plot(rewards_episodes)
# plt.figure(2)
# plt.plot(network_loss, 'o')
# plt.title('Training loss history')
# plt.xlabel('Iteration')
# plt.ylabel('Training loss')
plt.show()
