import sys
sys.path.append("/Users/abhiram/Dropbox/Adaptive-systems/Test Cases Algorithms/")
from pond_single_test import pond
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import swmm
import random
import seaborn


model = Sequential()
model.add(Dense(10, init='lecun_uniform', input_shape=(2, )))
model.add(Activation('relu'))

model.add(Dense(10, init='lecun_uniform'))
model.add(Activation('relu'))

model.add(Dense(2, init='lecun_uniform'))
model.add(Activation('softmax'))

model.load_weights('Neural_pond.h5')


Episodes = 200
POND_TEST = pond(100.0, 2.0)
POND_TEST.timestep = 1
ACTION_SPACE = ['1.0','0.0']

for i in range(0, Episodes):
    # Initialize new episodes
    POND_TEST.volume = 0
    POND_TEST.overflow = 0
    EPSILON = 0.7

    time = 0

    while POND_TEST.overflow == 0:
        time = time + 1
        if time > 5000:
            break
        observation = [POND_TEST.height, qin]
        q_values = model.predict(observation, batch_size=1)
        #--------- INFLOW -------#
        qin = 2.00
        if np.random.rand(1) < EPSILON:
            action = random.choice(ACTION_SPACE)
        else:
            action = np.argmax(q_values)
        






