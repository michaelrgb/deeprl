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

def conv2d(x, W, stride=1, padding='VALID'):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
def max_pool(x, size=4, stride=1, padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                          strides=[1, stride, stride, 1], padding=padding)
def lrelu(x):
    return tf.maximum(x, x*1e-3)

def weight_variable(shape, init_zeros=False):
    return tf.Variable(initial_value=tf.zeros(shape) if init_zeros else tf.truncated_normal(shape, stddev=0.1, seed=0))

def imshow(imlist):
    import matplotlib.pyplot as plt
    kwargs = {'interpolation': 'nearest'}
    plt.close()
    imlist = wrapList(imlist)
    f, axarr = plt.subplots(len(imlist))
    #axarr = wrapList(axarr)# Not list if only 1 image
    for i, nparray in enumerate(imlist):
        shape = nparray.shape
        if shape[-1] == 1:
            # If its greyscale then remove the 3rd dimension if any
            nparray = nparray.reshape((shape[0], shape[1]))
            # Plot negative pixels on the blue channel
            kwargs['cmap'] = 'bwr'
            kwargs['vmin'] = -1.
            kwargs['vmax'] = 1.
        ax = axarr[i] if len(imlist) > 1 else axarr
        im = ax.imshow(nparray, **kwargs)
        f.colorbar(im, ax=ax)
    f.show()

def gaussian_filter(kernel_size):
    x = np.zeros((kernel_size, kernel_size, 1, 1), dtype=DTYPE.name)

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = np.floor(kernel_size / 2.)
    for i in xrange(0, kernel_size):
        for j in xrange(0, kernel_size):
            x[i, j, 0, 0] = gauss(i - mid, j - mid)

    weights = x / np.sum(x)
    return tf.constant(weights, DTYPE)

def local_contrast_norm(x, gaussian_weights):
    # Move the input channels into the batches
    shape = tf.shape(x)
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [shape[0]*shape[3], shape[1], shape[2], 1])

    # For each pixel, remove local mean
    mean = conv2d(x, gaussian_weights, padding='SAME')
    mean_subtracted = x - mean

    # Calculate local standard deviation
    local_stddev = tf.sqrt(conv2d(mean_subtracted**2, gaussian_weights, padding='SAME'))

    # Lower gives more noise in areas of low contrast, i.e. non-edges
    threshold = 1e-1

    # Divide by the local stddev, with threshold to prevent divide-by-0
    local_stddev = tf.maximum(local_stddev, threshold)
    x = mean_subtracted / local_stddev

    # Rescale to [0 1]
    x = tf.maximum(x, 0.)
    x /= tf.maximum(tf.reduce_max(x, axis=[1, 2], keep_dims=True), threshold)

    # Restore the input channels
    x = tf.reshape(x, [shape[0], shape[3], shape[1], shape[2]])
    x = tf.transpose(x, [0, 2, 3, 1])
    return x

def layer_conv(x, conv_width, conv_stride, input_channels, output_channels):
    with tf.name_scope('weights'):
        W_conv = weight_variable([conv_width, conv_width, input_channels, output_channels])
        variable_summaries(W_conv)
    with tf.name_scope('bias'):
        b_conv = weight_variable([output_channels])
        variable_summaries(b_conv)
    x = wrapList(x)
    x = [local_contrast_norm(i, GAUSS_W) for i in x]
    x = [conv2d(i, W_conv, stride=conv_stride) + b_conv for i in x]
    return x, [W_conv, b_conv]

def make_conv(x, chan_in):
    conv_weights = []
    chan_out = 32
    for l in range(3):
        with tf.name_scope('layer%i' % l):
            x, w = layer_conv(x, 5, 2, chan_in, chan_out)
            conv_weights += w
            chan_in = chan_out; chan_out *= 2
    return x, conv_weights

def layer_reshape_flat(x, conv_eval):
    input_size = conv_eval.shape[1]
    input_channels = conv_eval.shape[3]
    flat_size = input_size * input_size * input_channels
    print('Convolution shape:', conv_eval.shape, 'resizing to flat:', flat_size)
    x = wrapList(x)
    x = [tf.reshape(i, [-1, flat_size]) for i in x]
    return x, flat_size

def layer_fully_connected(x, flat_size, outputs, activation=lrelu):
    with tf.name_scope('weights'):
        W_f = weight_variable([flat_size, outputs])
        variable_summaries(W_f)
    with tf.name_scope('bias'):
        b_f = weight_variable([outputs])
        variable_summaries(b_f)
    x = wrapList(x)
    x = [tf.matmul(i, W_f) + b_f for i in x]
    if activation:
        x = [activation(i) for i in x]
    return x, [W_f, b_f]

def layer_linear_sum(x, inputs, outputs=None, init_zeros=False):
    with tf.name_scope('weights'):
        W_f = weight_variable([inputs, outputs or 1], init_zeros)
        variable_summaries(W_f)
    x = wrapList(x)
    x = [tf.matmul(i, W_f) for i in x]
    if outputs is None:
        # Remove dimension
        x = [i[:, 0] for i in x]
    return x, [W_f]

def layer_make_features(x, conditions):
    x = wrapList(x)
    x = [[tf.expand_dims(tf.cast(cond(i), DTYPE), 1) for cond in conditions] for i in x]
    return [tf.concat(i, axis=1) for i in x]
