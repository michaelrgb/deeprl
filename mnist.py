#!/usr/bin/python

import tensorflow as tf, numpy as np
from utils import *

SAVE_SUMMARIES = False
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
def next_batch(batch_size, group=mnist.train):
    batch = group.next_batch(batch_size)
    batch = [batch[0].reshape([-1, 28, 28, 1]), batch[1]]
    return batch

sess = tf.InteractiveSession()
def init_vars(): sess.run(tf.global_variables_initializer())

x_image = tf.placeholder(DTYPE, shape=[None, 28, 28, 1])
y_truth = tf.placeholder(DTYPE, shape=[None, 10])
dropout_keep = tf.placeholder_with_default(1.0, shape=())

x = x_image

chan_in, chan_out = 1, 32
[x], w = layer_conv(x, 7, 1, chan_in, chan_out)
x = max_pool(x, size=3)

chan_in = chan_out; chan_out *= 2
[x], w = layer_conv(x, 7, 1, chan_in, chan_out)
x = max_pool(x, size=3)

chan_in = chan_out; chan_out *= 2
[x], w = layer_conv(x, 7, 1, chan_in, chan_out)
x = max_pool(x, size=3)

init_vars()
batch = next_batch(1)
[x], flat_size = layer_reshape_flat(x, x.eval(feed_dict={x_image: batch[0]}))

HIDDEN_LAYERS = 2
HIDDEN_NODES = 1024

for i in range(HIDDEN_LAYERS):
    x = tf.nn.dropout(x, keep_prob=dropout_keep)
    [x] = layer_fully_connected(x, flat_size, HIDDEN_NODES, activation=tf.nn.selu)
    flat_size = HIDDEN_NODES

x = tf.nn.dropout(x, keep_prob=dropout_keep)
[y_out] = layer_fully_connected(x, HIDDEN_NODES, 10, activation=None) # unscaled logits

LEARNING_RATE = 1e-3
opt = tf.train.AdamOptimizer(LEARNING_RATE, epsilon=0.1)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_truth, logits=y_out))
grad = opt.compute_gradients(cross_entropy, var_list=None)
train_step = opt.apply_gradients(grad)

correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_truth, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, DTYPE))
y_out_prob = tf.nn.softmax(y_out)

if SAVE_SUMMARIES:
    LOG_DIR = '/tmp/train'
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.summary.scalar('accuracy', accuracy)
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

def test_image():
    import Image
    im = Image.open('test_mnist.png')
    im = np.array(im)/255.
    im = im.reshape([28, 28, 1])
    prediction, argmax = sess.run([y_out_prob, tf.argmax(y_out_prob, 1)], feed_dict={x_image: [im]})
    for i in range(10):
        print(i, '{0:.5f}'.format(prediction[0][i]))
    print('argmax', argmax)

init_vars()
for i in range(1000):
    batch = next_batch(5000)
    feed_dict = {x_image: batch[0], y_truth: batch[1], dropout_keep: 0.5}
    acc_eval, _ = sess.run([accuracy, train_step], feed_dict=feed_dict)

    if 0:
        im, im_norm = sess.run([x_image, local_contrast_norm(x_image, GAUSS_W)], feed_dict=feed_dict)
        imshow([im[0], im_norm[0]])

    if SAVE_SUMMARIES:
        summary = sess.run([summaries], feed_dict=feed_dict)
        writer.add_summary(summary, i)

    print(dict(batch=i, accuracy=acc_eval))

batch = next_batch(10000, mnist.test)
acc_eval = sess.run(accuracy, feed_dict={x_image: batch[0], y_truth: batch[1]})
print(dict(test_accuracy=acc_eval))

test_image()
