import gym
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt


env=gym.make('CartPole-v0')

model = Sequential()
model.add(Dense(100, init='lecun_uniform', input_shape=(4,)))
model.add(Activation('relu'))

model.add(Dense(100, init='lecun_uniform'))
model.add(Activation('relu'))

model.add(Dense(2, init='lecun_uniform'))
model.add(Activation('relu'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

EPISODES = 1000
GAMMA = 0.8
EPSILON = 0.9
avg_reward = []
steps = []

for i in range(EPISODES):
    obs = env.reset()
    obs = np.reshape(obs, (1, 4))
    done = False
    j = 0
    rew = []
    while j < 1000:
        EPSILON = 0.99*EPSILON
        env.render()
        qval = model.predict(obs, batch_size=1)
        if (np.random.randn(1) < EPSILON):  # choose random action
            action = env.action_space.sample()
        else:
            action = np.argmax(qval)
        observation, reward, done, _ = env.step(action)
        if done:
            break
        observation = np.reshape(observation, (1, 4))
        qnew = model.predict(observation, batch_size=1)
        maxq_new = np.max(qnew)
        y = np.zeros((1, 2))
        y[:] = qval[:]
        update = (reward + (GAMMA * maxq_new))
        y[0][action] = update
        model.fit(observation, y, batch_size=1, nb_epoch=1, verbose=1)
        obs = observation
        j = j + 1
        rew.append(reward)
    steps.append(j)
    avg_reward.append(np.mean(rew))

plt.figure(1)
plt.plot(avg_reward)
plt.title('Average Reward')
plt.figure(2)
plt.plot(steps)
plt.title('Steps')
plt.show()
