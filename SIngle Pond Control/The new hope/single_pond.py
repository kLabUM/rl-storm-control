from pond_single_test import pond
import copy
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import seaborn

# Reward Function
def reward(height):
    """Reward Function"""
    if height > 1.10 and height <= 1.20:
        return 0.0
    elif height < 1.10:
        return (height-1.10)
    else:
        return  1.20 - height*1.50


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


def gate(action, current_gate):
    "Gate position"
    alpha = 0.1
    if action == 1:  # Open
        gate_pos = current_gate + alpha
        gate_pos = min(gate_pos, 1.0)
    else:  # Close
        gate_pos = current_gate - alpha
        gate_pos = max(gate_pos, 0.0)
    return gate_pos


def build_network():
    """Neural Nets Action-Value Function"""
    model = Sequential()
    model.add(Dense(10, input_dim=1))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('linear'))
    sgd = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


# Pond Testing
# Pond with 100 Sq.m and 2.0 m Max Height
single_pond = pond(100.0, 2.0)
single_pond.timestep(1.0)

# Q-estimator
model = build_network()
target_net = build_network()
model.load_weights('single_pond_new_test3.h5')
target_net.set_weights(model.get_weights())

# Replay Memory
states = np.zeros((1, 1))
rewards = np.zeros((1, 1))
action_replay = np.zeros((1, 1))
states_n = np.zeros((1, 1))
terminal = np.zeros((1, 1))


window_length = 100000
episode_count = 0
time = 0
steps = 50000
epsilon = np.linspace(0.01, 0.001, steps+10)
#  Book Keeping
rewards_episodes = []
network_loss = []
height_pond_mean = []
# Simulation

while time < steps:
    print("time", time)
    reward_tracker = []
    action_tracker = []
    height_pond = []
    episode_count += 1
    done = False
    train = False#if time > 1000 else False
    single_pond.height = 0
    single_pond.volume = 0
    loss_check = 0
    episode_time = 0
    qin = 2.0
    action_space = np.linspace(0.0, 1.0, 2)
    observation = np.array([[single_pond.height]])
    current_gate = 0.0
    while episode_time < 500 and not(done):

        if single_pond.height > 1.90:  # Check for terminal step
            done = True
        else:
            done = False

        episode_time += 1
        time += 1

        state_step = observation
        states = snapshots(states, state_step, window_length)

        q_values = model.predict(state_step)
        height_pond.append(single_pond.height)
        action = epsi_greedy(action_space, q_values,
                             epsilon[min(time, steps)])
        action_tracker.append(current_gate)
        action_replay = snapshots(action_replay, action, window_length)
        gate_position = gate(action, current_gate)
        single_pond.dhdt(qin, single_pond.qout(gate_position))
        current_gate = copy.deepcopy(gate_position)

        temp_terminal = np.zeros((1, 1))
        temp_terminal[0][0] = done
        terminal = snapshots(terminal, temp_terminal, window_length)

        reward_step = reward(single_pond.height)
        rewards = snapshots(rewards, reward_step, window_length)
        reward_tracker.append(reward_step)

        observation = np.array([[single_pond.height]])
        state_step_n = observation
        states_n = snapshots(states_n, state_step_n, window_length)

        if train:
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

    rewards_episodes.append(np.mean(reward_tracker))
    height_pond_mean.append(np.mean(height_pond))

for layer in model.layers:
    weights = layer.get_weights()
    print(weights)

model.save_weights('single_pond_new_test311.h5')
plt.figure(1)
plt.plot(rewards_episodes)
plt.title('Mean Rewards')
plt.figure(2)
plt.plot(height_pond)
plt.title('Height Last Episode')
plt.figure(3)
plt.plot(action_tracker)
plt.title('action tracker')
plt.figure(4)
plt.plot(height_pond_mean)
plt.title('Mean Height')
plt.show()
