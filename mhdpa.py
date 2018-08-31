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
    def __init__(self, heads=4, d_k=32, d_ff=128):
        super(MHDPA, self).__init__()
        self.heads = heads
        self.d_k = d_k
        self.d_ff = d_ff

    def apply(self, x):
        s = x.shape.as_list()
        d_v = s[-1]
        if not self.weights:
            # Sublayer 1
            self.query = self.add_variable('query', [d_v, self.heads, self.d_k])
            self.key = self.add_variable('key',     [d_v, self.heads, self.d_k])
            self.value = self.add_variable('value', [d_v, self.heads, d_v])
            self.final = self.add_variable('final', [self.heads, d_v, d_v])
            # Sublayer 2
            self.kernel1 = self.add_variable('kernel1', [d_v, self.d_ff])
            self.kernel2 = self.add_variable('kernel2', [self.d_ff, d_v])

        def layer_norm(n): return tf.contrib.layers.layer_norm(n, False, False, begin_norm_axis=-1)

        # Project each entity into q,k,v vectors
        query = tf.tensordot(x, self.query, [[-1], [0]])
        key = tf.tensordot(x, self.key, [[-1], [0]])
        value = tf.tensordot(x, self.value, [[-1], [0]])

        if 0:
            query = layer_norm(query)
            key = layer_norm(key)
            value = layer_norm(value)

        # Compare each q with every other entity k via dot-product
        query = tf.expand_dims(tf.expand_dims(query, 3), 3)
        key = tf.expand_dims(tf.expand_dims(key, 1), 1)
        value = tf.expand_dims(tf.expand_dims(value, 1), 1)
        unnormalized = tf.reduce_sum(query * key, -1)

        # Softmax on combined dimension
        unnormalized = tf.reshape(unnormalized, [s[0], s[1]*s[2], s[1],s[2], self.heads])
        attention = tf.nn.softmax(unnormalized/self.d_k**0.5, 1)
        attention = tf.reshape(attention, [s[0], s[1],s[2], s[1],s[2], self.heads])

        # Weighted sum of attention values
        A = tf.expand_dims(attention, -1) * value
        A = tf.reduce_sum(tf.reduce_sum(A, 3), 3)

        # Concatenate and once again project, resulting in the final values
        final = tf.tensordot(A, self.final, [[-2,-1], [0,1]])

        # Residual&Norm 1
        sublayer1 = layer_norm(final+x)

        # 2-layer MLP
        mlp = tf.nn.relu(tf.tensordot(sublayer1, self.kernel1, [[-1], [0]]))
        mlp = tf.tensordot(mlp, self.kernel2, [[-1], [0]])

        # Residual&Norm 2
        sublayer2 = layer_norm(mlp+sublayer1)
        return sublayer2, attention
