import tensorflow as tf, numpy as np
from utils import *
sess = tf.InteractiveSession()

import gym
env = gym.make('CarRacing-v0')
INPUT_DIM = [96, 96, 3]
def make_layer(x, conv_width, input_channels, output_channels):
    with tf.name_scope('weights'):
        W_conv = weight_variable([conv_width, conv_width, input_channels, output_channels])
        variable_summaries(W_conv)
    with tf.name_scope('bias'):
        b_conv = bias_variable([output_channels])
        variable_summaries(b_conv)
    x = [lrelu(conv2d(i, W_conv) + b_conv) for i in x]
    x = [max_pool(i, size=3, stride=2) for i in x]
    return x

def make_net(x):
    with tf.name_scope('layer0'):
        x = make_layer(x, 5, INPUT_DIM[-1], 32)
    with tf.name_scope('layer1'):
        x = make_layer(x, 5, 32, 64)
    with tf.name_scope('layer2'):
        x = make_layer(x, 5, 64, 128)
    return x

def make_layer_flat(x, input_size, input_channels, output_channels):
    flat_size = input_size * input_size * input_channels
    with tf.name_scope('layer_flat'):
        with tf.name_scope('weights'):
            W_f = weight_variable([flat_size, output_channels])
            variable_summaries(W_f)
        with tf.name_scope('bias'):
            b_f = bias_variable([output_channels])
            variable_summaries(b_f)
        x = [tf.reshape(i, [-1, flat_size]) for i in x]
        x = [lrelu(tf.matmul(i, W_f) + b_f) for i in x]
    return x

state_tf = tf.placeholder(DTYPE, shape=[None] + INPUT_DIM)
state_next_tf = tf.placeholder(DTYPE, shape=[None] + INPUT_DIM)
reward_tf = tf.placeholder(DTYPE, shape=[None, 1])
net = make_net([state_tf, state_next_tf])

# Calculate topmost convolution dimensionality
sess.run(tf.global_variables_initializer())
observation = env.reset()
feed_dict = {state_tf: [observation]}
eval = net[0].eval(feed_dict)
[q_tf, q_next_tf] = make_layer_flat(net, input_size=eval.shape[1], input_channels=eval.shape[3], output_channels=1)

with tf.name_scope('loss'):
    # SARSA
    GAMMA = 0.9
    q_next_tf = tf.stop_gradient(q_next_tf)
    loss_tf = (reward_tf + GAMMA*q_next_tf - q_tf)
    loss_tf = loss_tf * loss_tf
    #loss_tf = tf.pow(loss_tf, 2)
    loss_sum = tf.reduce_sum(loss_tf)

    loss_argmin = tf.argmin(loss_tf, 0)

    tf.summary.histogram('histogram', loss_tf)
    tf.summary.scalar('sum', loss_sum)

ER_SIZE = 50
#er_state = tf.get_variable("er_state", [ER_SIZE] + INPUT_DIM) # FIXME: use variables later
#er_reward = tf.get_variable("er_reward", [ER_SIZE])
er_state = np.zeros([ER_SIZE] + INPUT_DIM)
er_state_next = np.zeros([ER_SIZE] + INPUT_DIM)
er_reward = np.zeros([ER_SIZE, 1])
# Random reward values to check we are actually modifying weights
#er_reward = np.random.standard_normal([ER_SIZE, 1])

sess.run(tf.global_variables_initializer())
feed_dict = {state_tf: er_state, state_next_tf: er_state_next, reward_tf: er_reward}
eval = q_tf.eval(feed_dict)# TEST

opt = tf.train.GradientDescentOptimizer(0.00000001)
#grads = opt.compute_gradients(loss_sum)
train_step = opt.minimize(loss_sum)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)

def gym_loop():
    observation = env.reset()
    state = observation

    frame = 0
    while 1:
        env.render('human')# current display
        #rgb_video = env.render('rgb_array')

        action = env.action_space.sample() # take a random action
        observation, reward, done, info = env.step(action)

        er_index = frame
        if frame >= ER_SIZE:
            summary, loss_sum_eval, loss_argmin_eval, _ = sess.run([merged, loss_sum, loss_argmin, train_step], feed_dict)
            er_index = loss_argmin_eval[0]
            print(loss_sum_eval, er_index)
            #loss_eval = sess.run(loss_tf, feed_dict); print(loss_eval)

            train_writer.add_summary(summary, frame)

        er_state[er_index] = state
        er_state_next[er_index] = observation
        er_reward[er_index] = reward


        state = observation
        frame += 1
gym_loop()
