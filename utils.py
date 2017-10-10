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

def conv2d(x, W, padding='VALID'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
def max_pool(x, size=4, stride=2, padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                          strides=[1, stride, stride, 1], padding=padding)
def lrelu(x):
    return tf.maximum(x, x*0.01)

def weight_variable(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape, stddev=0.1), dtype=DTYPE)
def bias_variable(shape):
    return weight_variable(shape)
    return tf.Variable(initial_value=tf.constant(0.1, shape=shape), dtype=DTYPE)

def imshow(nparray):
    import matplotlib.pyplot as plt
    kwargs = {'interpolation': 'nearest'}
    shape = nparray.shape
    if shape[-1] == 1:
        # If its greyscale then remove the 3rd dimension if any
        nparray = nparray.reshape((shape[0], shape[1]))
        # Plot negative pixels on the blue channel
        kwargs['cmap'] = 'bwr'
        kwargs['vmin'] = -1.
        kwargs['vmax'] = 1.
    plt.close()
    plt.imshow(nparray, **kwargs)
    plt.colorbar()
    plt.show()
