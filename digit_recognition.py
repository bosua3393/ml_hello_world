import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
from numpy import round

n_input = 784
n_hidden1 = 300
n_hidden2 = 300
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


correct_pred = tf.equal(tf.argmax(y3, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


saver = tf.train.Saver()

from image import my_image
test = my_image

with tf.Session() as sess:
    saver.restore(sess, "./best_model/model.ckpt")
    output = sess.run(y3, {x: test})
    for row in range(28):
        print(test[0][row*28:(row+1)*28])
    print(f'digit recognized: {np.argmax(output[0])}')
    '''
    print("testing:")
    test_x, test_label = mnist.test.next_batch(100000)
    print(sess.run(accuracy, {x: test_x, label: test_label}))
    # print(round(w2.eval()))
    '''
