import tensorflow as tf, numpy as np
DTYPE = tf.float32

def wrapList(value):
    return value if type(value) == list else [value]

def variable_summaries(var):
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.scalar('max', tf.reduce_max(var))
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.histogram('histogram', var)

def conv2d(x, W, padding='VALID'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
def max_pool(x, size=4, stride=2, padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                          strides=[1, stride, stride, 1], padding=padding)
def lrelu(x):
    return tf.maximum(x, x*0.01)

def weight_variable(shape, init_zeros=False):
    return tf.Variable(initial_value=tf.zeros(shape) if init_zeros else tf.truncated_normal(shape, stddev=0.1))

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
