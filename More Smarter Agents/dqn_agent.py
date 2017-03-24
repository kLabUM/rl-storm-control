import numpy as np


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
                 states,
                 replay_memory,
                 policy,
                 batch_size=32,
                 target_update=10000,
                 train=True):

        self.states = states
        self.ac_model = action_value_model
        self.target_model = target_model
        self.replay = replay_memory
        self.batch_size = batch_size
        self.policy = policy
        self.train = train
        self.target_update = target_update

        self.state_vector = np.zeros((1, self.states))
        self.state_new_vector = np.zeros((1, self.states))
        self.rewards_vector = np.zeros((1))
        self.terminal_vector = np.zeros((1))
        self.action_vector = np.zeros((1))

        self.training_batch = {'states': np.zeros((self.batch_size,
                                                   self.states)),
                               'states_new': np.zeros((self.batch_size,
                                                       self.states)),
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
        q_values_train_next = self.target_model.predict_on_batch(
            temp_states_new)
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

    def train_q(self, timesteps):
        self._random_sample()
        temp = True if timesteps > 1000 else False
        if temp:
            self._update_target_model()
        self._train()

    def actions_q(self, epsilon, action_space):
        q_values = self.ac_model.predict(self.state_vector)
        action = self.policy(action_space, q_values, epsilon)
        return action
