import tensorflow as tf
from tensorflow.python.layers import base

# Concat x,y coordinate onto each entity feature vector
def concat_coord_xy(x):
    s = x.shape.as_list()
    f = (s[1]-1)/2.
    coord_x = (tf.range(s[1], dtype='float32')-f)/f
    coord_x = tf.expand_dims(tf.expand_dims([coord_x], 1), -1)
    coord_x = tf.tile(coord_x, [1, x.shape[2].value, 1, 1])
    f = (s[2]-1)/2.
    coord_y = (tf.range(s[2], dtype='float32')-f)/f
    coord_y = tf.expand_dims(tf.expand_dims([coord_y], 2), -1)
    coord_y = tf.tile(coord_y, [1, 1, x.shape[1].value, 1])

    coord = tf.concat([coord_y, coord_x], -1)
    coord = tf.tile(coord, [s[0], 1, 1, 1])
    return tf.concat([x, coord], -1)

class MHDPA(base.Layer):
    def __init__(self, heads=4, d=32):
        super(MHDPA, self).__init__()
        self.heads = heads
        self.d = d

    def apply(self, x, combine_heads=True):
        s = x.shape.as_list()
        if not self.weights:
            shape = [s[-1], self.heads, self.d]
            self.query = self.add_variable('query', shape)
            self.key = self.add_variable('key', shape)
            self.value = self.add_variable('value', shape)
            self.kernel1 = self.add_variable('kernel1', shape[1:] + [s[-1]])
            self.kernel2 = self.add_variable('kernel2', [s[-1]]*2)

        # Project each entity into q,k,v vectors
        query = tf.tensordot(x, self.query, [[-1], [0]])
        key = tf.tensordot(x, self.key, [[-1], [0]])
        value = tf.tensordot(x, self.value, [[-1], [0]])

        # Normalize q,k,v vectors using layer normalization
        query = tf.contrib.layers.layer_norm(query, False, False, begin_norm_axis=-1)
        key = tf.contrib.layers.layer_norm(key, False, False, begin_norm_axis=-1)
        value = tf.contrib.layers.layer_norm(value, False, False, begin_norm_axis=-1)

        # Compare each q with every other entity k via dot-product
        query = tf.expand_dims(tf.expand_dims(query, 3), 3)
        key = tf.expand_dims(tf.expand_dims(key, 1), 1)
        value = tf.expand_dims(tf.expand_dims(value, 1), 1)
        unnormalized = tf.reduce_sum(query * key, -1)

        # Softmax on combined dimension
        unnormalized = tf.reshape(unnormalized, [s[0], s[1]*s[2], s[1],s[2], self.heads])
        attention = tf.nn.softmax(unnormalized/self.d**0.5, 1)
        attention = tf.reshape(attention, [s[0], s[1],s[2], s[1],s[2], self.heads])

        # Weighted sum of attention values
        A = tf.expand_dims(attention, -1) * value
        A = tf.reduce_sum(tf.reduce_sum(A, 3), 3)

        if not combine_heads:
            A = tf.contrib.layers.layer_norm(A, False, False, begin_norm_axis=-1)
            return A, attention

        # Pass A vectors through the 2-layer MLP
        mlp = tf.nn.relu(tf.tensordot(A, self.kernel1, [[3,4], [0,1]]))
        mlp = tf.nn.relu(tf.tensordot(mlp, self.kernel2, [[-1], [0]]))

        # Sum mlp with original input and normalize
        x = tf.contrib.layers.layer_norm(x+mlp, False, False, begin_norm_axis=-1)
        return x, attention
