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

model1.load_weights('no_hope_p1.h5')
model2.load_weights('no_hope_p2.h5')

swmm.initialize(inp)
reward_tracker_pond1 = []
action_tracker_pond1 = []
outflow_tracker_pond1 = []
outflow_tracker_pond2 = []
reward_tracker_pond2 = []
action_tracker_pond2 = []

height_pond1_tracker = []
height_pond2_tracker = []

outflow_tracker = []

episode_time = 0
while episode_time < 7200:
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
    outflow_tracker_pond1.append(swmm.get('R2', swmm.FLOW, swmm.SI))
    outflow_tracker_pond2.append(swmm.get('C8', swmm.FLOW, swmm.SI))
    # Book Keeping
    height_pond1_tracker.append(height_pond1)
    height_pond2_tracker.append(height_pond2)
    outflow_tracker.append(outflow)

    swmm.modify_setting('R2', action_pond1/100.0)
    swmm.modify_setting('C8', action_pond2/100.0)

    swmm.run_step()


swmm.initialize(inp)

height1 = []
outflow1 = []
height2 = []
outflow2 = []
rain = []
qout = []
t = 0
while t < 7200:
    swmm.run_step()
    rain.append(swmm.get('SC2', swmm.PRECIPITATION, swmm.SI))
    height1.append(swmm.get('S5', swmm.DEPTH, swmm.SI))
    outflow1.append(swmm.get('R2', swmm.FLOW, swmm.SI))
    height2.append(swmm.get('S7', swmm.DEPTH, swmm.SI))
    outflow2.append(swmm.get('C8', swmm.FLOW, swmm.SI))
    qout.append(swmm.get('C8', swmm.FLOW, swmm.SI))
    t = t + 1

print 'Mass Balance Check :', np.sum(outflow_tracker_pond2)-np.sum(outflow2)

plt.figure(1)
plt.subplot(4, 2, 1)
plt.plot(rain)
plt.ylabel('Rainfall [in]')

plt.subplot(4, 2, 3)
plt.gca().set_color_cycle([[0.05, 0.56, 0.92], 'black'])
plt.plot(height_pond1_tracker, label='Controlled')
plt.plot(height1, label='Uncontrolled', linestyle='--')
plt.ylabel("Pond Height [m]")
plt.title('Pond-1')

plt.subplot(4, 2, 4)
plt.gca().set_color_cycle([[0.47, 0.67, 0.18], 'black'])
plt.plot(height_pond2_tracker, label='Controlled')
plt.plot(height2, label='Uncontrolled', linestyle='--')
plt.title('Pond-2')

plt.subplot(4, 2, 5)
plt.gca().set_color_cycle([[0.05, 0.56, 0.92], 'black'])
plt.plot(outflow_tracker_pond1, label='Controlled')
plt.plot(outflow1, label='Uncontrolled', linestyle='--')
plt.ylabel('Discharge [cu.m/sec]')

plt.subplot(4, 2, 6)
plt.gca().set_color_cycle([[0.47, 0.67, 0.18], 'black'])
plt.plot(outflow_tracker_pond2, label='Controlled')
plt.plot(outflow2, label='Uncontrolled', linestyle='--')

plt.subplot(4, 2, 7)
plt.gca().set_color_cycle([[0.05, 0.56, 0.92]])
plt.plot(action_tracker_pond1)
plt.ylabel('Gate Position [%]')
plt.xlabel('Time Steps')
plt.subplot(4, 2, 8)
plt.gca().set_color_cycle([[0.47, 0.67, 0.18]])
plt.plot(action_tracker_pond2)
plt.xlabel('Time Steps')
plt.show()
