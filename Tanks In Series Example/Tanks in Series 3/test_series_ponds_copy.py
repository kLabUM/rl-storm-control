import swmm
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import seaborn as sns
sns.set_palette("RdBu_r")
# Grid
sns.set_style("white")
# Font and Font Size
csfont = {'font': 'Helvetica',
          'size': 14}
plt.rc(csfont)

def build_network():
    """Neural Nets Action-Value Function"""
    model = Sequential()
    model.add(Dense(10, input_dim=3))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(101))
    model.add(Activation('linear'))
    sgd = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

# Ponds Testing
inp = 's.inp'

# Q-estimator for Pond 1
model1 = build_network()


# Q-estimator for Pond 2
model2 = build_network()

model1.load_weights('pond1_4.h5')
model2.load_weights('pond2_4.h5')

swmm.initialize(inp)
reward_tracker_pond1 = []
action_tracker_pond1 = []

reward_tracker_pond2 = []
action_tracker_pond2 = []

height_pond1_tracker = []
height_pond2_tracker = []

outflow_tracker = []

episode_time = 0
while episode_time < 3000:
    episode_time += 1
    height_pond1 = swmm.get('S5', swmm.DEPTH, swmm.SI)
    height_pond2 = swmm.get('S7', swmm.DEPTH, swmm.SI)

    inflow_pond1 = swmm.get('S5', swmm.INFLOW, swmm.SI)
    inflow_pond2 = swmm.get('S7', swmm.INFLOW, swmm.SI)
    outflow = swmm.get('C8', swmm.FLOW, swmm.SI)

    observation_pond1 = np.array([[height_pond1,
                                   height_pond2,
                                   inflow_pond1]])

    observation_pond2 = np.array([[height_pond1,
                                   height_pond2,
                                   inflow_pond2]])

    q_values_pond1 = model1.predict(observation_pond1)
    q_values_pond2 = model2.predict(observation_pond2)

    action_pond1 = np.argmax(q_values_pond1)
    action_pond2 = np.argmax(q_values_pond2)

    action_tracker_pond1.append(action_pond1/100.0)
    action_tracker_pond2.append(action_pond2/100.0)

    # Book Keeping
    height_pond1_tracker.append(height_pond1)
    height_pond2_tracker.append(height_pond2)
    outflow_tracker.append(outflow)

    swmm.modify_setting('R2', action_pond1/100.0)
    swmm.modify_setting('C8', action_pond2/100.0)

    swmm.run_step()


swmm.initialize(inp)

height1 = []
height2 = []
rain = []
qout = []
t = 0
while t < 3000:
    swmm.run_step()
    rain.append(swmm.get('SC2', swmm.PRECIPITATION, swmm.SI))
    height1.append(swmm.get('S5', swmm.DEPTH, swmm.SI))
    height2.append(swmm.get('S7', swmm.DEPTH, swmm.SI))
    qout.append(swmm.get('C8', swmm.FLOW, swmm.SI))
    t = t + 1

plt.figure(1)
plt.subplot(2, 3, 1)
plt.plot(height_pond1_tracker, label='Controlled')
plt.plot(height1, label='Uncontrolled')
plt.title('Height - Pond 1')
plt.legend()
plt.subplot(2, 3, 2)
plt.plot(height_pond2_tracker, label='Controlled')
plt.plot(height2, label='Uncontrolled')
plt.title('Height - Pond 2')
plt.legend()
plt.subplot(2, 3, 3)
plt.plot(outflow_tracker, label='Controlled')
plt.plot(qout, label='Uncontrolled')
plt.title('Outflow - System')
plt.legend()
plt.subplot(2, 3, 4)
plt.plot(action_tracker_pond1)
plt.title('Actions - Pond 1')
plt.subplot(2, 3, 5)
plt.plot(action_tracker_pond2)
plt.title('Actions - Pond 2')
plt.subplot(2, 3, 6)
plt.plot(rain)
plt.show()
