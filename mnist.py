#!/usr/bin/python

import tensorflow as tf, numpy as np
from utils import *

USE_MNIST = 0
if USE_MNIST:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
else:
    import generate
    training_data = [np.load(generate.FILE_IMAGES), np.load(generate.FILE_LABELS)]
def next_batch(batch_size):
    if USE_MNIST:
        batch = mnist.train.next_batch(batch_size)
        batch = [batch[0].reshape([-1, 28, 28, 1]), batch[1]]
    else:
        num_samples = training_data[0].shape[0]
        indices = np.random.permutation(num_samples)[:batch_size]
        batch = [training_data[0][indices], training_data[1][indices]]
    return batch

LOG_DIR = '/tmp/tflogdir'
if tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)

sess = tf.InteractiveSession()

x_image = tf.placeholder(DTYPE, shape=[None, 28, 28, 1])
y = tf.placeholder(DTYPE, shape=[None, 10])

GAUSS_W = gaussian_filter(5)

if 0:
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y_out = tf.matmul(x, W) + b
else:
    layer_x = x_image
    layer_x = local_contrast_norm(layer_x, GAUSS_W)

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(layer_x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_pool1 = local_contrast_norm(h_pool1, GAUSS_W)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_out = tf.matmul(h_fc1, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_out, y))

train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, DTYPE))

tf.summary.scalar('accuracy', accuracy)
image_test = x_image[0]
image_test = tf.reshape(image_test, [1,28,28,1]) # summary.image() requires 4D tensor
#tf.summary.image('first_mnist', image_test)

summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(500):
    batch = next_batch(1000)
    feed_dict = {x_image: batch[0], y: batch[1]}
    train_step.run(feed_dict=feed_dict)

    if 0:
        imshow(x_image.eval(feed_dict)[0])
        x_image_norm = local_contrast_norm(x_image, GAUSS_W)
        imshow(x_image_norm.eval(feed_dict)[0])

    summary, acc_eval = sess.run([summaries, accuracy], feed_dict=feed_dict)

    writer.add_summary(summary, i)

    print('Batch', i, 'Accuracy: ', acc_eval)
