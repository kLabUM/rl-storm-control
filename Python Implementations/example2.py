from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf

# Parameters
learning_rate = 0.001
episodes = 15
batch_size = 100
display_step = 1

# Network Parameters
nodes_hl1 = 50  # nodes hidden layer 1
nodes_hl2 = 50  # nodes hidden layer 2
n_input = 784  # 28*28
n_classes = 10  # 10 digits

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float"m [None, n_classes])


def network(x, weights, biases):
    """Network Structure"""
    # Layer 1
    hl1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    hl1 = tf.nn.crelu(hl1)
    # Layer 2
    hl2 = tf.add(tf.matmul(hl1, weights['h2']), biases['b2'])
    hl2 = tf.nn.crelu(hl2)
    # Output Layer
    out_layer = tf.matmul(hl2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, nodes_hl1])),
    'h2': tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2])),
    'out': tf.Variable(tf.random_normal([nodes_hl2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([nodes_hl1])),
    'b2': tf.Variable(tf.random_normal([nodes_hl2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Build Model

net = network(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net , y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
