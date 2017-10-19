GAMMA = 0.9
STATE_FRAMES = 4
ER_SIZE = 100
GRADIENT_CORRECTION = True
STATE_FEATURE_SIZE = 1000
LEARNING_RATE_LINEAR = 1e-6/STATE_FEATURE_SIZE
LEARNING_RATE_CONV = 1e-10

SAVE_SUMMARIES = False
RENDER_ENV = False
SHOW_OBS = False
MAX_EPISODE_STATES = 50
TEST_ENV = False
if TEST_ENV:
    ER_SIZE = 1
    STATE_FRAMES = 1
    MAX_EPISODE_STATES = 2

import tensorflow as tf, numpy as np
from utils import *

if TEST_ENV:
    FRAME_DIM = [1]
    ACTION_DIM = 1
    class TestEnv:
        def reset(self):
            self.state = [1]
            return self.state
        def render(self): pass
        def step(self, action):
            self.state = [2]# Only initial and second state
            reward = 0.
            done = False
            return self.state, reward, done, {}
    env = TestEnv()
else:
    FRAME_DIM = [96, 96, 3]
    ACTION_DIM = 3
    import gym
    env = gym.make('CarRacing-v0')

STATE_DIM = FRAME_DIM[:]
STATE_DIM[-1] *= STATE_FRAMES
env.reset(); env.render()# Gym seems to need at least 1 reset&render before we get valid observation

def layer_conv(x, conv_width, input_channels, output_channels):
    with tf.name_scope('weights'):
        W_conv = weight_variable([conv_width, conv_width, input_channels, output_channels])
        variable_summaries(W_conv)
    with tf.name_scope('bias'):
        b_conv = weight_variable([output_channels])
        variable_summaries(b_conv)
    x = wrapList(x)
    x = [lrelu(conv2d(i, W_conv) + b_conv) for i in x]
    x = [max_pool(i, size=5, stride=2) for i in x]

    return x, [W_conv, b_conv]

def layer_reshape_flat(x, conv_eval):
    input_size = conv_eval.shape[1]
    input_channels = conv_eval.shape[3]
    flat_size = input_size * input_size * input_channels
    x = wrapList(x)
    x = [tf.reshape(i, [-1, flat_size]) for i in x]
    return x, flat_size

def make_net(x):
    if TEST_ENV:
        # No convolution for test env
        return x, []

    with tf.name_scope('layer0'):
        x, conv_weights = layer_conv(x, 5, STATE_DIM[-1], 32)
    with tf.name_scope('layer1'):
        x, w = layer_conv(x, 5, 32, 64)
        conv_weights += w
    with tf.name_scope('layer2'):
        x, w = layer_conv(x, 5, 64, 128)
        conv_weights += w
    return x, conv_weights

def layer_fully_connected(x, flat_size, outputs):
    with tf.name_scope('weights'):
        W_f = weight_variable([flat_size, outputs])
        variable_summaries(W_f)
    with tf.name_scope('bias'):
        b_f = weight_variable([outputs])
        variable_summaries(b_f)
    x = wrapList(x)
    x = [lrelu(tf.matmul(i, W_f) + b_f) for i in x]
    return x, [W_f, b_f]

def layer_linear_sum(x, flat_size, outputs, init_zeros):
    with tf.name_scope('weights'):
        W_f = weight_variable([flat_size, outputs], init_zeros)
        variable_summaries(W_f)
    x = wrapList(x)
    x = [tf.matmul(i, W_f) for i in x]
    return x, W_f

tf.set_random_seed(0)

sess = tf.InteractiveSession()
with tf.name_scope('experience_replay'):
    er_state =      tf.Variable(tf.zeros([ER_SIZE] + STATE_DIM),    False, name='state')
    er_next_state = tf.Variable(tf.zeros([ER_SIZE] + STATE_DIM),    False, name='next_state')
    er_action =     tf.Variable(tf.zeros([ER_SIZE] + [ACTION_DIM]), False, name='action')
    er_reward =     tf.Variable(tf.zeros([ER_SIZE, 1]),             False, name='reward')

if TEST_ENV:
    state_features = [er_state, er_next_state]
    STATE_FEATURE_SIZE = STATE_DIM[-1]
else:
    conv_net, conv_weights = make_net([er_state, er_next_state])
    # Calculate topmost convolution dimensionality to create fully-connected layer
    sess.run(tf.global_variables_initializer())
    conv_net, conv_flat_size = layer_reshape_flat(conv_net, conv_net[0].eval())
    state_features, w = layer_fully_connected(conv_net, conv_flat_size, STATE_FEATURE_SIZE)
    conv_weights += w

with tf.name_scope('q_value'):
    [q_tf, q_next_tf], q_weights = layer_linear_sum(state_features, STATE_FEATURE_SIZE, 1, False)

with tf.name_scope('td_error'):
    # SARSA
    td_error = er_reward + GAMMA*tf.stop_gradient(q_next_tf) - q_tf

    td_error_sq = td_error**2
    td_error_loss = td_error_sum = tf.reduce_sum(td_error_sq)
    loss_argmin = tf.argmin(td_error_sq, 0)
    tf.summary.scalar('loss', td_error_sum)

with tf.name_scope('td_approx'):
    if GRADIENT_CORRECTION:
        [td_approx], td_approx_weights = layer_linear_sum(state_features[0], STATE_FEATURE_SIZE, 1, True)
        td_approx_sum = tf.reduce_sum((td_error - td_approx)**2)
        tf.summary.scalar('loss', td_approx_sum)
    else:
        td_approx_sum = tf.constant(0.)

sess.run(tf.global_variables_initializer())

# Use auto-differentiation just for convolutional weights
opt_conv = tf.train.GradientDescentOptimizer(LEARNING_RATE_CONV)
opt_approx = tf.train.GradientDescentOptimizer(LEARNING_RATE_LINEAR)
opt_td = tf.train.GradientDescentOptimizer(LEARNING_RATE_LINEAR/100)
q_apply_ops = []
if not TEST_ENV:
    grads = opt_conv.compute_gradients(td_error_loss, var_list=conv_weights)
    q_apply_ops += [opt_conv.apply_gradients(grads)]

def reshape_grad(x): return tf.reshape(-tf.reduce_sum(x, axis=0), [STATE_FEATURE_SIZE, 1])
grad_q = td_error*state_features[0]
approx_apply_op = []
if GRADIENT_CORRECTION:
    # TD error with gradient correction (TDC)
    grad_q += -GAMMA*td_approx*state_features[1]

    # Learn to approximate the TD error
    grad = (td_error - td_approx)*state_features[0]
    approx_apply_op += [opt_approx.apply_gradients([(reshape_grad(grad), td_approx_weights)])]
q_apply_ops += [opt_td.apply_gradients([(reshape_grad(grad_q), q_weights)])]

if SAVE_SUMMARIES:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)

def rl_loop():
    state = np.zeros([STATE_FRAMES] + FRAME_DIM, 'uint8')
    next_state = np.empty_like(state)

    episode = 0
    er_num_saves = 0
    while 1:
        # New episode
        observation = env.reset()

        for ep_state in range(MAX_EPISODE_STATES):
            if TEST_ENV:
                action = [0.]
            else:
                action = env.action_space.sample() # take a random action
                action = [0, 1, 0]

            next_reward = 0.
            for frame in range(STATE_FRAMES):
                if SHOW_OBS:
                    print('Frame', ep_state, frame); imshow(observation)
                if RENDER_ENV:
                    env.render()# Not needed for state_pixels

                next_state[frame] = observation
                observation, reward, done, info = env.step(action)
                next_reward += reward

            if ep_state > 0:
                if er_num_saves < ER_SIZE:
                    save_index = er_num_saves
                else:
                    ops = q_apply_ops + approx_apply_op
                    if SAVE_SUMMARIES:
                        ops += [merged]
                    r = sess.run([loss_argmin, td_error_sum, td_approx_sum] + ops)
                    save_index=r[0][0]
                    save_index = er_num_saves % ER_SIZE
                    r = dict(
                        episode=episode,
                        save_index=save_index,
                        td_error_sum=r[1],
                        td_approx_sum=r[2])
                    print(r)
                    #loss_eval = sess.run(td_error); print(loss_eval)
                    if SAVE_SUMMARIES:
                        summary = r[3]
                        train_writer.add_summary(summary, er_num_saves)

                sess.run([er_state[save_index].assign(state.reshape(STATE_DIM)/255.),
                         er_next_state[save_index].assign(next_state.reshape(STATE_DIM)/255.),
                         er_reward[save_index].assign([reward_sum])])
                er_num_saves += 1

            state, next_state = next_state, state
            reward_sum = next_reward

        episode += 1

rl_loop()
