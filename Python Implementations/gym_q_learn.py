import tensorflow as tf
import numpy as np
import gym

# Parameters
learning_rate = 0.001
display_step = 1

# Network Size
hl1 = 50
hl2 = 50
n_input = 4
n_output = 2

# Place holders
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_output])

# Create model

def network(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Define Action Space

def action_space(value):
    """Action Defining From Variable In Neural Network"""
    if value == 0:
        return 1
    else:
        return 0

# Store layers weight & bias


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, hl1])),
    'h2': tf.Variable(tf.random_normal([hl1, hl2])),
    'out': tf.Variable(tf.random_normal([hl2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([hl1])),
    'b2': tf.Variable(tf.random_normal([hl2])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Construct model
prediction = network(x, weights, biases)
action_take = tf.argmax(prediction, 1)


nextQ = tf.placeholder(shape=[None], dtype=tf.float32, name="q1")
Qw = tf.placeholder(shape=[None], dtype=tf.float32, name="q2")
loss = tf.squared_difference(nextQ, Qw)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
trainer = optimizer.minimize(loss)
num_episodes = 1
epsilon = 0.5
ALPHA = 0.01
GAMMA = 0.6

# Rewards and Steps Per Episodes
rewads_episode = []
steps_episode = []

# Open AI gym environment, Cart Problem
env = gym.make('CartPole-v0')

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        obs = env.reset()
        j = 0
        while j < 100:
            env.render()

            obs = np.reshape(obs, (1, 4))
            Q_values = s.run([prediction], feed_dict={x: obs})
            temp = Q_values[0]

            a = np.argmax(Q_values[0])
            if np.random.randn(1) < epsilon:
                a = env.action_space.sample()
            Q_value_old = temp[0, a]

            #  Get new state and reward from environment
            observation, reward, done, _ = env.step(a)
            if done:
                break

            # Next State --> With action a
            observation = np.reshape(observation, (1, 4))
            Q_new_values = s.run([prediction], feed_dict={x: observation})
            maxQ_new = np.max(Q_new_values)

            # Parameters Update
            Q = (reward + GAMMA*maxQ_new)

            # Training
            q, w = s.run([trainer, loss], feed_dict={nextQ: Q, Qw: Q_value_old})
            obs = observation
            j += 1


