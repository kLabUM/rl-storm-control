import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import swmm
import random
import seaborn

# Reward Function


def reward_environment(height, outflow):
    """Reward Function"""
    if height > 3.90 and height < 4.00:
        reward = 1
    elif height > 4.5:
        reward = -100
    else:
        reward = -1
    return reward
# Neural Network


model = Sequential()
model.add(Dense(10, init='lecun_uniform', input_shape=(3, )))
model.add(Activation('relu'))

model.add(Dense(10, init='lecun_uniform'))
model.add(Activation('relu'))

model.add(Dense(2, init='lecun_uniform'))
model.add(Activation('softmax'))

sgd = SGD()
model.compile(loss='mse', optimizer=sgd)

# SWMM Test Case
INP = 'Testcase.inp'  # Test-case SWMM
action_space = [1, 0]  # Open and Close


# Reinforcement Learning Parameters
EPISODES = 200  # Number of Episodes
GAMMA = 0.5  # Discount Factor
EPSILON = 0.6  # Exploration Rate

# Observers
average_reward_episodes = []  # Average Reward For Episode
steps_epsiode = []  # Time Steps Per Episode
depth_epsiode = []  # Depth

# Implementation R-Learning
for i in range(EPISODES):

    swmm.initialize(INP)  # Initialize SWMM Test case

    t = 0  # Time of Simulation
    reward_step = []  # Reward For Individual Steps
    depth_step = []
    inflow_step = []
    outflow_step = []
    volume_step = []
    action_step = []

    #  Extract States from SWMM
    precipitation = swmm.get('S1', swmm.PRECIPITATION, swmm.SI)  # Rain
    depth = swmm.get('S3', swmm.DEPTH, swmm.SI)  # Depth in tank
    inflow = swmm.get('C3', swmm.FLOW, swmm.SI)  # Inflow into tank
    flow_catchment = swmm.get('J1', swmm.INFLOW, swmm.SI)  # Outflow Catch

    inflow_step.append(inflow)

    # Observation for the Net
    observation = np.array([depth, inflow, flow_catchment])
    observation = np.reshape(observation, (1, 3))

    while t < 400:
        EPSILON = 0.9*EPSILON

        # Predict Response from Net
        q_values = model.predict(observation, batch_size=1)

        # Take Action based on Epsilon Greedy
        # Action Space = [open, close] of outlet
        if (np.random.randn(1) < EPSILON):  # Epsilon Greedy
            action = random.choice(action_space)
        else:
            action = np.argmax(q_values)

        # Implement Action
        swmm.modify_setting('R1', action)
        swmm.run_step()

        # Observations
        depth_1 = swmm.get('S3', swmm.DEPTH, swmm.SI)  # Depth in tank
        outflow_1 = swmm.get('C1', swmm.FLOW, swmm.SI)  # Outflow from tank
        inflow_1 = swmm.get('C3', swmm.FLOW, swmm.SI)  # Inflow into tank
        flow_catchment_1 = swmm.get('J1', swmm.INFLOW, swmm.SI)  # Out-Catch
        volume_1 = swmm.get('S3', swmm.VOLUME, swmm.SI)  # Volume

        observation_1 = np.array([depth_1, inflow_1, flow_catchment_1])
        observation_1 = np.reshape(observation_1, (1, 3))

        # Reward
        reward = reward_environment(depth_1, outflow_1)

        # Prediction for next state
        q_values_1 = model.predict(observation_1, batch_size=1)
        q_max_1 = np.max(q_values_1)

        # Loss Function
        y = np.zeros((1, 2))
        y_1 = np.zeros((1, 2))
        y[:] = q_values[:]
        y_1[0][0] = y[0][action]
        update = (reward + (GAMMA * q_max_1))
        y_1[0][1] = update

        # Train Network
        model.fit(observation, y_1, batch_size=1, nb_epoch=1, verbose=0)
        observation = observation_1

        # Additional Updates
        t = t + 1
        outflow_step.append(outflow_1)
        reward_step.append(reward)
        depth_step.append(depth_1)
        volume_step.append(volume_1)
        action_step.append(action)
        inflow_step.append(inflow_1)
    print i
    average_reward_episodes.append(np.mean(reward_step))
    depth_epsiode.append(np.mean(depth_step))

model.save('Neural_pond1_reward.h5')

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(average_reward_episodes)
plt.title('Average Reward')
plt.subplot(2, 1, 2)
plt.plot(depth_epsiode)
plt.title('Depth')

plt.figure(2)
plt.subplot(4, 1, 1)
#plt.plot(precipitation, linewidth=2.0, label='Precipitation')
plt.plot(inflow_step, linewidth=3.0, label='Inflow')
plt.title('Pond Visualization')
plt.legend(loc='upper right', fontsize='small')

plt.subplot(4, 1, 2)
plt.plot(depth_step, linewidth=2.0, label='No Control')
plt.ylabel('Depth')

plt.subplot(4, 1, 3)
#plt.plot(outflow_step, linewidth=2.0, label='Outflow')
plt.plot(volume_step, linewidth=3.0, label='Volume')
plt.legend(loc='upper right', fontsize='small')

plt.subplot(4, 1, 4)
plt.plot(action_step)
plt.ylabel('% Opening')
plt.show()
