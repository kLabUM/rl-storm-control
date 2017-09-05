import numpy as np


class replay_stacker():
    def __init__(self, columns, window_length=100):
        self._data = np.zeros((window_length, columns))
        self.capacity = window_length
        self.size = 0
        self.columns = columns

    def update(self, x):
        self._add(x)

    def _add(self, x):
        if self.size == self.capacity:
            self._data = np.roll(self._data, -1)
            self._data[self.size-1, :] = x
        else:
            self._data[self.size, :] = x
            self.size += 1

    def data(self):
        return self._data[0:self.size, :]


class replay_memory_agent():
    def __init__(self, states_len, replay_window):
        self.states_len = states_len
        self.replay_window = replay_window

        # Initialize replay memory
        self.replay_memory = {'states': replay_stacker(self.states_len,
                                                       self.replay_window),
                              'states_new': replay_stacker(self.states_len,
                                                           self.replay_window),
                              'rewards': replay_stacker(1,
                                                        self.replay_window),
                              'actions': replay_stacker(1,
                                                        self.replay_window),
                              'terminal': replay_stacker(1,
                                                         self.replay_window)}

    def replay_memory_update(self,
                             states,
                             states_new,
                             rewards,
                             actions,
                             terminal):
        self.replay_memory['rewards'].update(rewards)
        self.replay_memory['states'].update(states)
        self.replay_memory['states_new'].update(states_new)
        self.replay_memory['actions'].update(actions)
        self.replay_memory['terminal'].update(terminal)

def randombatch(sample_size, replay_size):
    indx = np.linspace(0, replay_size-1, sample_size)
    indx = np.random.choice(indx, sample_size, replace=False)
    indx.tolist()
    indx = list(map(int, indx))
    return indx


class deep_q_agent:
    def __init__(self,
                 action_value_model,
                 target_model,
                 states_len,
                 replay_memory,
                 policy,
                 batch_size=32,
                 target_update=10000,
                 train=True):

        self.states_len = states_len
        self.ac_model = action_value_model
        self.target_model = target_model
        self.replay = replay_memory
        self.batch_size = batch_size
        self.policy = policy
        self.train = train
        self.target_update = target_update

        self.state_vector = np.zeros((1, self.states_len))
        self.state_new_vector = np.zeros((1, self.states_len))
        self.rewards_vector = np.zeros((1))
        self.terminal_vector = np.zeros((1))
        self.action_vector = np.zeros((1))

        self.training_batch = {'states': np.zeros((self.batch_size,
                                                   self.states_len)),
                               'states_new': np.zeros((self.batch_size,
                                                       self.states_len)),
                               'actions': np.zeros((self.batch_size, 1)),
                               'rewards': np.zeros((self.batch_size, 1)),
                               'terminal': np.zeros((self.batch_size, 1))}

    def _random_sample(self):
        indx = randombatch(self.batch_size, len(self.replay['states'].data()))
        for i in self.training_batch.keys():
            temp = self.replay[i].data()
            self.training_batch[i] = temp[indx]

    def _update_target_model(self):
        self.target_model.set_weights(self.ac_model.get_weights())

    def _train(self):
        temp_states_new = self.training_batch['states_new']
        temp_states = self.training_batch['states']
        temp_rewards = self.training_batch['rewards']
        temp_terminal = self.training_batch['terminal']
        temp_actions = self.training_batch['actions']
        q_values_train_next = self.target_model.predict_on_batch(temp_states_new)
        target = self.ac_model.predict_on_batch(temp_states)
        for i in range(self.batch_size):
            action_idx = int(temp_actions[i])
            if temp_terminal[i]:
                target[i][action_idx] = temp_rewards[i]
            else:
                target[i][action_idx] = temp_rewards[i] + 0.99 * np.max(
                    q_values_train_next[i])

        self.ac_model.fit(temp_states,
                          target,
                          batch_size=32,
                          nb_epoch=1,
                          verbose=0)

    def train_q(self, update):
        self._random_sample()
        if update:
            self._update_target_model()
        self._train()

# Policy Function
def epsi_greedy(action_space, q_values, epsilon):
    """Epsilon Greedy"""
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    else:
        return np.argmax(q_values)

