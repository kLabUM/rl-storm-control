from fc_net import FullyConnectedNet
from solver import Solver
import copy
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
import gym


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


# Neural Net Parameters
weight_scale = 1
learning_rate = 0.0001

# Q-estimator
model = FullyConnectedNet([50],
                          input_dim=4,
                          num_classes=2,
                          weight_scale=weight_scale,
                          dtype=np.float64)

# Target estimator
target_model = FullyConnectedNet([50],
                                 input_dim=4,
                                 num_classes=2,
                                 weight_scale=weight_scale,
                                 dtype=np.float64)

# Replay Memory
states = np.zeros((1, 4))
rewards = np.zeros((1, 1))
action_replay = np.zeros((1, 1))
states_n = np.zeros((1, 4))
terminal = np.zeros((1, 1))

# Environment Testing
action_space = np.linspace(0, 1, 2, dtype=int)
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, "/Users/abhiram/Desktop/Open Ai Test/")

# Reinforcement Learning Parameters and Book Keeping
window_length = 50000
episode_count = 0
time = 0
steps = 150000
epsilon = np.linspace(0.4, 0.0, 130000+10)
rewards_episodes = []
network_loss = []

# Simulation
while time < steps:
    observation = env.reset()
    observation = np.reshape(observation, (1, 4))
    reward_tracker = []
    episode_count += 1
    done = False
    train = True if time > 1000 else False
    loss_check = 0
    while not(done):
        time += 1
        state_step = observation
        states = snapshots(states, state_step, window_length)
        #  Loss with out y computes the scores from the network
        q_values = model.loss(state_step)

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

        if train:
            indx = randombatch(32, states.shape[0])
            indx = list(map(int, indx))
            states_train = states[indx]
            reward_train = rewards[indx]
            states_n_train = states_n[indx]
            action_train = action_replay[indx]
            terminal_train = terminal[indx]

            if time % 8000 == 0:
                target_model.params = copy.deepcopy(model.params)

            q_values_train_next = target_model.loss(states_n_train)

            target = model.loss(states_train)

            for i in range(32):
                action_idx = int(action_train[i])

                if terminal_train[i]:
                    target[i][action_idx] = reward_train[i]
                else:
                    target[i][action_idx] = reward_train[i] + 0.99 * np.max(q_values_train_next[i])

            small_data = {'X_train': states_train, 'y_train': target}

            solver = Solver(model,
                            small_data,
                            print_every=1,
                            num_epochs=1,
                            batch_size=32,
                            verbose=False,
                            update_rule='sgd',
                            optim_config={'learning_rate': learning_rate})

            solver.train()
            loss_check += np.sum(solver.loss_history)
    network_loss.append(loss_check)
    rewards_episodes.append(np.sum(reward_tracker))

for i in model.params:
    print(model.params[i])
env.close()
gym.upload("/Users/abhiram/Desktop/Open Ai Test/", api_key='sk_tHy7ubR2T7mu1KmejCddyA')
plt.figure(1)
plt.plot(rewards_episodes)
plt.figure(2)
plt.plot(network_loss, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()
