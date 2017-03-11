import numpy as np
import swmm


class numpy_list:
    def __init__(self, column_length):
        self.data = np.zeros((100, column_length))
        self.capacity = len(self.data)
        self.size = 0
        self.column_length = column_length

    def update(self, x):
        self.add(x)

    def add(self, x):
        if self.size == self.capacity:
            self.capacity *= 4
            newdata = np.zeros((self.capacity, self.column_length))
            newdata[0:self.size, :] = self.data[0:self.size, :]
            self.data = newdata

        self.data[self.size, :] = x
        self.size += 1


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


# Policy Epsilon Greedy
def epsi_greedy(action_space, q_values, epsilon):
    """Espilon - Greedy"""
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    else:
        return np.argmax(q_values)


# Reward Function
def reward(overflow, outflow):
    """Reward Function"""
    if outflow <= 0.20:
        reward_out = 10.0*outflow
    else:
        reward_out = -outflow*10.0
    out = np.sum(overflow)
    return reward_out + out*10.00


# SWMM Network finder
def swmm_network(Network, state):
    temp = []
    for i in Network:
        temp.append(swmm.get(i, state, swmm.SI))
    temp = np.asarray(temp)
    return temp
