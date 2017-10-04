import tensorflow as tf, numpy as np
DTYPE = tf.float32

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
def weight_variable(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape, stddev=0.1))
def bias_variable(shape):
    return weight_variable(shape)
    return tf.Variable(initial_value=tf.constant(0.1, shape=shape))
