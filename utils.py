import tensorflow as tf, numpy as np
DTYPE = tf.float32

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool(x, size=2, stride=2):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                          strides=[1, stride, stride, 1], padding='SAME')
def lrelu(x):
    return tf.maximum(x, x*0.000001)

def weight_variable(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape, stddev=0.1))
def bias_variable(shape):
    return weight_variable(shape)
    return tf.Variable(initial_value=tf.constant(0.1, shape=shape))

