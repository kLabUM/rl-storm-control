from pond_single_test import pond
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import seaborn as sns
import copy
#born Plots Configuration
# Palette
sns.set_palette("Paired")
# Grid
sns.set_style("whitegrid")
# Font and Font Size
csfont = {'font': 'Helvetica',
          'size': 14}
plt.rc(csfont)


def gate(action, current_gate):
    "Gate position"
    alpha = 0.1
    if action == 1:  # Open
        gate_pos = current_gate + alpha
        gate_pos = min(gate_pos, 1.0)
    else:  # Close
        gate_pos = current_gate - alpha
        gate_pos = max(gate_pos, 0.0)
    return gate_pos

def build_network():
    """Neural Nets Action-Value Function"""
    model = Sequential()
    model.add(Dense(10, input_dim=1))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('linear'))
    sgd = SGD(lr=0.01, clipnorm=1.)
    model.compile(loss='mse', optimizer=sgd )
    return model


# Pond Testing
# Pond with 100 Sq.m and 2.0 m Max Height
single_pond = pond(100.0, 2.0)
single_pond.timestep(1.0)

# Q-estimator
model = build_network()
model.load_weights('single_pond_new_test3.h5')
summary_heights = np.zeros((1, 1000))
summary_actions = np.zeros((1, 1000))
# Controlled Example
for i in range(1, 8):
    time = 0
    height_controlled = []
    actions_control = []
    single_pond.height = 0
    single_pond.volume = 0
    current_gate = 0.0
    while time < 1000:
        time += 1
        state = np.array([[single_pond.height]])
        action = np.argmax(model.predict(state))
        gate_position = gate(action, current_gate)
        current_gate = copy.deepcopy(gate_position)
        single_pond.dhdt(i, single_pond.qout(gate_position))
        height_controlled.append(single_pond.height)
        actions_control.append(gate_position)
    height_controlled = np.array(height_controlled).reshape(1, 1000)
    actions_control = np.array(actions_control).reshape(1, 1000)
    summary_heights = np.vstack((summary_heights, height_controlled))
    summary_actions = np.vstack((summary_actions, actions_control))

# Uncontrolled Case
summary_heights_uncontrolled = np.zeros((1, 1000))
for i in range(1, 8):
    time = 0
    height_controlled = []
    actions_control = []
    single_pond.height = 0
    single_pond.volume = 0
    while time < 1000:
        time += 1
        single_pond.dhdt(i, single_pond.qout(1.0))
        height_controlled.append(single_pond.height)
        actions_control.append(gate_position)
    height_controlled = np.array(height_controlled).reshape(1, 1000)
    summary_heights_uncontrolled = np.vstack((summary_heights_uncontrolled, height_controlled))


plt.subplot(2, 1, 1)
for i in range(0, summary_heights.shape[0]):
    plt.plot(summary_heights[i], label= 'Inflow :' + str(i))
plt.ylabel('Depth')
plt.title('Controlled Response for Various Inflows')
plt.legend()
plt.subplot(2, 1, 2)
for i in range(0, summary_heights_uncontrolled.shape[0]):
    plt.plot(summary_heights_uncontrolled[i])
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Depth')
plt.title('Uncontrolled')

# plt.subplot(3, 1, 3)
#for i in range(0, summary_actions.shape[0]):
#    plt.plot(summary_actions[i], label=str(i))

#plt.xlabel('Time Steps')
#plt.ylabel('Gate Opening')
plt.show()

