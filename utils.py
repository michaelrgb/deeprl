import tensorflow as tf, numpy as np
DTYPE = tf.float32

def loop_while(f):
    while f(): pass
def wrapList(value):
    return value if type(value) == list else [value]
class Struct:
    def __init__(self, **entries): self.__dict__.update(entries)
def softmax(x):
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()

def variable_summaries(var, scope=None):
    if not scope: scope = var.name.split('/')[-1].split(':')[0]
    with tf.name_scope(scope):
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', var)

def tf_gradients(cost, weights): return zip(tf.gradients(cost, weights), weights)

def conv2d(x, W, stride=1, padding='VALID'):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
def max_pool(x, size=4, stride=1, padding='VALID'):
    x = wrapList(x)
    return [tf.nn.max_pool(i, ksize=[1, size, size, 1],
                           strides=[1, stride, stride, 1], padding=padding) for i in x]

def weight_variable(shape, init_zeros=False):
    return tf.Variable(initial_value=tf.zeros(shape) if init_zeros else
        tf.contrib.layers.xavier_initializer(seed=0)(shape))
def scope_vars(scope_name='', GLOBAL=False): # '' for current scope
    current = tf.get_variable_scope().name
    if current: scope_name = current + '/' + scope_name
    key = tf.GraphKeys.GLOBAL_VARIABLES if GLOBAL else tf.GraphKeys.TRAINABLE_VARIABLES
    return tf.get_collection(key, scope=scope_name)
def grads_clamp(grads, max_value):
    grads = wrapList(grads)
    return [(tf.where(w > max_value, tf.abs(g), tf.where(w < -max_value, -tf.abs(g), g)), w) for g,w in grads]
def grads_index(grads, name_substr):
    return [i for i in range(len(grads)) if name_substr in grads[i][1].name]

def accum_value(value, ops_add, ops_clear):
    accum = tf.Variable(tf.zeros_like(value), trainable=False)
    ops_add.append(accum.assign_add(value))
    ops_clear.append(accum.assign(tf.zeros_like(value)))
    return accum

def imshow(imlist):
    import matplotlib.pyplot as plt
    kwargs = {'interpolation': 'nearest'}
    if type(imlist) == np.ndarray and len(imlist.shape) == 4:
        imlist = [imlist[i] for i in range(imlist.shape[0])]
    else:
        imlist = wrapList(imlist)
    f, axarr = plt.subplots(ncols=len(imlist))
    #axarr = wrapList(axarr)# Not list if only 1 image
    for i, nparray in enumerate(imlist):
        shape = nparray.shape
        if shape[-1] == 1:
            # If its grayscale then remove the 3rd dimension if any
            nparray = nparray.reshape((shape[0], shape[1]))
            # Plot negative pixels on the blue channel
            kwargs['cmap'] = 'bwr'
            kwargs['vmin'] = -1.
            kwargs['vmax'] = 1.
        ax = axarr[i] if len(imlist) > 1 else axarr
        im = ax.imshow(nparray, **kwargs)
        f.colorbar(im, ax=ax)
    plt.show() # Wait until close

def gaussian_filter(kernel_size):
    x = np.zeros((kernel_size, kernel_size, 1, 1), dtype=DTYPE.name)

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = np.floor(kernel_size / 2.)
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x[i, j, 0, 0] = gauss(i - mid, j - mid)

    weights = x / np.sum(x)
    return tf.constant(weights, DTYPE)

def local_contrast_norm(x, gaussian_weights, scale01=False):
    # Move the input channels into the batches
    shape = x.shape.as_list()
    x = tf.transpose(x, [0, 3, 1, 2])
    x = tf.reshape(x, [shape[0]*shape[3], shape[1], shape[2], 1])

    # For each pixel, remove local mean
    mean = conv2d(x, gaussian_weights, padding='SAME')
    mean_subtracted = x - mean

    # Calculate local standard deviation
    local_stddev = tf.sqrt(tf.maximum(0., conv2d(mean_subtracted**2, gaussian_weights, padding='SAME')))

    # Lower gives more noise in areas of low contrast, i.e. non-edges
    threshold = 1e-1

    # Divide by the local stddev, with threshold to prevent divide-by-0
    local_stddev = tf.maximum(local_stddev, threshold)
    x = mean_subtracted / local_stddev

    if scale01:
        # Rescale to [0 1]
        x = tf.maximum(x, 0.)
        x /= tf.maximum(tf.reduce_max(x, axis=[1, 2], keep_dims=True), threshold)

    # Restore the input channels
    x = tf.reshape(x, [shape[0], shape[3], shape[1], shape[2]])
    x = tf.transpose(x, [0, 2, 3, 1])
    return x

GAUSS_W = gaussian_filter(5)
def test_lcn(image, sess):
    frame_ph = tf.placeholder(tf.float32, [None] + list(image.shape[-3:]))
    lcn_op = local_contrast_norm(frame_ph, GAUSS_W)
    lcn = sess.run(lcn_op, feed_dict={frame_ph: image if len(image.shape)==4 else [image]})
    return lcn

def finite_diff(y_diff, x_diff):
    no_div0 = tf.abs(x_diff) > 1e-2
    zeros = tf.zeros_like(x_diff)
    return tf.where(no_div0, tf.expand_dims(y_diff,-1) / x_diff, zeros)

@tf.RegisterGradient("CastGrad")
def _cast_grad(op, grad):
    return grad
@tf.RegisterGradient("GreaterGrad")
def _greater_grad(op, grad):
    x = op.inputs[0]
    # During backpropagation heaviside behaves like sigmoid
    return tf.sigmoid(x) * (1 - tf.sigmoid(x)) * grad, None

def heaviside(x, g=tf.get_default_graph()):
    with g.gradient_override_map({"Greater": "GreaterGrad", "Cast": "CastGrad"}):
        return tf.cast(x>0, DTYPE)
