import tensorflow as tf
from numpy import round

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_loop = 10000
learn_rate = 10
batch_size = 1000

n_input = 784
n_hidden1 = 16
n_hidden2 = 16
n_output = 10

x = tf.placeholder(tf.float32, [None, n_input])
label = tf.placeholder(tf.float32, [None, n_output])

with tf.name_scope(name='layer1'):
    w1 = tf.Variable(tf.random_uniform((n_input, n_hidden1), -1, 1), name='weight1')
    b1 = tf.Variable(tf.random_uniform((1, n_hidden1), -1, 1), name='bias1')
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1, name='output1')

with tf.name_scope(name='layer2'):
    w2 = tf.Variable(tf.random_uniform((n_hidden1, n_hidden2), -1, 1), name='weight2')
    b2 = tf.Variable(tf.random_uniform((1, n_hidden2), -1, 1), name='bias2')
    y2 = tf.nn.relu(tf.matmul(y1, w2) + b2, name='output2')

with tf.name_scope(name='output_layer'):
    w3 = tf.Variable(tf.random_uniform((n_hidden2, n_output), -1, 1), name='weight3')
    b3 = tf.Variable(tf.random_uniform((1, n_output), -1, 1), name='bias3')
    y3 = tf.nn.softmax(tf.matmul(y2, w3) + b3, name='output3')

loss = tf.losses.sigmoid_cross_entropy(label, y3)
train_method = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(y3, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #writer = tf.summary.FileWriter('./graphs3', graph=tf.get_default_graph())
    batch_x, batch_label = mnist.train.next_batch(batch_size)
    for step in range(train_loop):
        sess.run(train_method, {x: batch_x, label: batch_label})
        if step % 200 == 0:
            print("Step:", step, " Loss:", sess.run(loss, {x: batch_x, label: batch_label}), " Accuracy:", sess.run(accuracy, {x: batch_x, label: batch_label}))

    print("testing:")
    for i in range(10):
        print('-----------------------------------')
        test_x, test_label = mnist.test.next_batch(1)
        print(round(sess.run(y3, {x: test_x, label: test_label})))
        print(test_label)


