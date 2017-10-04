import tensorflow as tf, numpy as np
from utils import *
sess = tf.InteractiveSession()

INPUT_DIM = [96, 96, 3]# gym

def make_layer(x, conv_width, input_channels, output_channels):
    W_conv = weight_variable([conv_width, conv_width, input_channels, output_channels])
    b_conv = bias_variable([output_channels])
    x = tf.nn.relu(conv2d(x, W_conv) + b_conv)
    x = max_pool_2x2(x)
    return x
def make_net(x):
    net = make_layer(x,   5, INPUT_DIM[-1], 32)
    net = make_layer(net, 5, 32, 64)
    net = make_layer(net, 5, 64, 128)
    return net
def make_layer_flat(x, input_size, input_channels, output_channels):
    flat_size = input_size * input_size * input_channels
    W_f = weight_variable([flat_size, output_channels])
    b_f = bias_variable([output_channels])
    x = tf.reshape(x, [-1, flat_size])
    x = tf.nn.relu(tf.matmul(x, W_f) + b_f)
    return x

import gym
env = gym.make('CarRacing-v0')

x_image = tf.placeholder(DTYPE, shape=[None] + INPUT_DIM)
y = tf.placeholder(DTYPE, shape=[None, 1])
net = make_net(x_image)
sess.run(tf.global_variables_initializer())

# Calculate topmost convolution dimensionality
observation = env.reset()
feed_dict = {x_image: [observation]}
eval = net.eval(feed_dict)
net = make_layer_flat(net, eval.shape[1], eval.shape[3], 1)

sess.run(tf.global_variables_initializer()) # Made more variables
eval = net.eval(feed_dict)# test

loss = tf.reduce_sum((y - net)**2)
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

def gym_loop():
    observation = env.reset()
    state = observation

    while 1:
        env.render('human')# current display
        #rgb_video = env.render('rgb_array')

        action = env.action_space.sample() # take a random action
        observation, reward, done, info = env.step(action)

        feed_dict={x_image: [state], y: [[reward]]}
        sess.run(train_step, feed_dict)

        loss_eval = loss.eval(feed_dict)
        print(reward, loss_eval)

        state = observation
gym_loop()
