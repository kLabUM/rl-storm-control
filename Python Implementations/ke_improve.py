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

obmodel.add(Dense(2))
model.add(Activation('softmax'))

sgd = SGsnapshot(snaps, observation, 10)


env.monitoD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd)

EPISODES = 5
GAMMA = 0.5
EPSILON = 0.9
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

        if (np.random.randn(1) < EPSILON):  # choose random action
            action = env.action_space.sample()
            print (action)
        else:
            action = np.argmax(qval)
            print (action)



        observation, reward, done, _ = env.step(action)
        observation = np.reshape(observation, (1, 4))

        qnew = model.predict(observation, batch_size=1)
        maxq_new = np.max(qnew)

        y = np.zeros((1, 2))
        y[:] = qval[:]
        if not done:
            update = (reward + (GAMMA * maxq_new))
        else:
            update = reward

        y[0][action] = update
        model.fit(observation, y, batch_size=1, nb_epoch=1, verbose=0)

        obs = observation

        j = j + 1
        rew.append(reward)
        if done:
            break
    print ('Episode :', i)
    print ('Steps :', j)
    #print ('Reward :', np.sum(rew))
    steps.append(j)
    avg_reward.append(np.sum(rew))

#model.save_weights('my_model_weights.h5')

#plt.figure(1)
#plt.plot(avg_reward)
#plt.title('Average Reward')
#plt.figure(2)
#plt.plot(steps)
#plt.title('Steps')
#plt.show()
