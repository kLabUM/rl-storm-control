import gym
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import SGD
import matplotlib.pyplot as plt


env=gym.make('CartPole-v0')
#env.monitor.start('/tmp/cartpole-experiment-1', force=True)

model = Sequential()
model.add(Dense(20, input_shape=(4,)))
model.add(Activation('relu'))

model.add(Dense(20))
model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

model.load_weights('my_model_weights.h5')

EPISODES = 50
GAMMA = 0.0
EPSILON = 0.0
avg_reward = []
steps = []

for i in range(EPISODES):
    obs = env.reset()
    obs = np.reshape(obs, (1, 4))
    done = False
    j = 0
    rew = []
    while j < 500:
        EPSILON = 0.99*EPSILON
        qval = model.predict(obs, batch_size=1)
        action = np.argmax(qval)
        observation, reward, done, _ = env.step(action)
        observation = np.reshape(observation, (1, 4))
        obs = observation
        j = j + 1
        rew.append(reward)
        if done:
            break
    steps.append(j)
    avg_reward.append(np.sum(rew))



plt.figure(1)
plt.plot(avg_reward)
plt.title('Average Reward')
plt.figure(2)
plt.plot(steps)
plt.title('Steps')
plt.show()
