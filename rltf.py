# coding=utf8

GAMMA = 0.9
STATE_FRAMES = 4
ER_SIZE = 1000
GRADIENT_CORRECTION = True
COPDAC = True
STATE_FEATURES = 1000
LEARNING_RATE_LINEAR = 1e-2
LEARNING_RATE_CONV = 1e-12
MOMENTUM = 0.9

SAVE_SUMMARIES = False
SHOW_OBS = False
MAX_EPISODE_STATES = 10000
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
            self.state = np.array([1.])
            return self.state
        def render(self): pass
        def step(self, action):
            self.state = np.array([2.])# Only initial and second state
            reward = 0.
            done = False
            return self.state, reward, done, {}
    env = TestEnv()
else:
    FRAME_DIM = [96, 96, 3]
    ACTION_DIM = 3
    import gym
    env = gym.make('CarRacing-v0')
    car_racing = gym.envs.box2d.car_racing
    car_racing.WINDOW_W = 800 # Default is huge
    car_racing.WINDOW_H = 600

STATE_DIM = FRAME_DIM[:]
STATE_DIM[-1] *= STATE_FRAMES

sess = tf.InteractiveSession()

GAUSS_W = gaussian_filter(5)
def test_lcn(image, batch_idx=0):
    frame_ph = tf.placeholder(tf.float32, [None] + list(image.shape[-3:]))
    lcn_op = local_contrast_norm(frame_ph, GAUSS_W)
    lcn = sess.run(lcn_op, feed_dict={frame_ph: np.expand_dims(image, 0)})
    return lcn[batch_idx]

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
    #x = [max_pool(i, size=5, stride=2) for i in x]
    #x = [tf.nn.dropout(i, keep_prob=0.5) for i in x]
    return x, [W_conv, b_conv]

def make_net(x):
    if TEST_ENV:
        # No convolution for test env
        return x, []

    with tf.name_scope('layer0'):
        chan_out = 32
        x, w = layer_conv(x, 5, 2, STATE_DIM[-1], chan_out)
        conv_weights = w
    with tf.name_scope('layer1'):
        chan_in = chan_out; chan_out *= 2
        x, w = layer_conv(x, 5, 2, chan_in, chan_out)
        conv_weights += w
    with tf.name_scope('layer2'):
        chan_in = chan_out; chan_out *= 2
        x, w = layer_conv(x, 5, 2, chan_in, chan_out)
        conv_weights += w
    return x, conv_weights

def layer_reshape_flat(x, conv_eval):
    input_size = conv_eval.shape[1]
    input_channels = conv_eval.shape[3]
    flat_size = input_size * input_size * input_channels
    print('Convolution shape:', conv_eval.shape, 'resizing to flat:', flat_size)
    x = wrapList(x)
    x = [tf.reshape(i, [-1, flat_size]) for i in x]
    return x, flat_size

def layer_fully_connected(x, flat_size, outputs):
    with tf.name_scope('weights'):
        W_f = weight_variable([flat_size, outputs])
        variable_summaries(W_f)
    with tf.name_scope('bias'):
        b_f = weight_variable([outputs])
        variable_summaries(b_f)
    x = wrapList(x)
    x = [tf.matmul(i, W_f) + b_f for i in x]
    # Reduce number of "active" features
    x = [tf.nn.softmax(i) for i in x]
    return x, [W_f, b_f]

def layer_linear_sum(x, inputs, outputs, init_zeros=False):
    with tf.name_scope('weights'):
        W_f = weight_variable([inputs, outputs], init_zeros)
        variable_summaries(W_f)
    x = wrapList(x)
    x = [tf.matmul(i, W_f) for i in x]
    return x, W_f

with tf.name_scope('experience_replay'):
    er_state =      tf.Variable(tf.zeros([ER_SIZE] + STATE_DIM),    False, name='state')
    er_next_state = tf.Variable(tf.zeros([ER_SIZE] + STATE_DIM),    False, name='next_state')
    er_action =     tf.Variable(tf.zeros([ER_SIZE, ACTION_DIM]),    False, name='action')
    er_reward =     tf.Variable(tf.zeros([ER_SIZE, 1]),             False, name='reward')

def init_vars(): sess.run(tf.global_variables_initializer())

state_ph = tf.placeholder(tf.float32, [None] + STATE_DIM)
with tf.name_scope('state_features'):
    state_features = [er_state, er_next_state, state_ph]
    if TEST_ENV:
        STATE_FEATURES = STATE_DIM[-1]
    else:
        conv_net, conv_weights = make_net(state_features)
        # Calculate topmost convolution dimensionality to create fully-connected layer
        init_vars()
        conv_net, conv_flat_size = layer_reshape_flat(conv_net, conv_net[0].eval())
        state_features, w = layer_fully_connected(conv_net, conv_flat_size, STATE_FEATURES)
        conv_weights += w

if COPDAC:
    with tf.name_scope('policy'):
        # a = μ_θ(s)
        [policy, policy_ph], policy_weights = layer_linear_sum([state_features[0], state_features[2]], STATE_FEATURES, ACTION_DIM)
        [policy_delta], w_weights = layer_linear_sum(state_features[0], STATE_FEATURES, ACTION_DIM)

# Qw(s, a) = (a − μ_θ(s)) T ∇_θ[μ_θ(s)] T w + V_v(s)
with tf.name_scope('state_value'):
    # Baseline value, independent of action a
    q_value, baseline_weights = layer_linear_sum(state_features, STATE_FEATURES, 1)
if COPDAC:
    with tf.name_scope('q_value'):
        # φ(s, a) = ∇_θ[μ_θ(s)](a − μ_θ(s))
        state_action_features = tf.expand_dims(state_features[0], 2) * tf.expand_dims(er_action - policy, 1)
        # Advantage of taking action a over action μ_θ(s)
        action_adv = tf.reduce_sum(state_action_features * tf.expand_dims(w_weights, 0), axis=[1, 2])
        action_adv = tf.expand_dims(action_adv, 1)
        # q_value[1] is by definition the same as V_v(s_t+1)
        q_value[0] += action_adv

with tf.name_scope('td_error'):
    td_error = er_reward + GAMMA*tf.stop_gradient(q_value[1]) - q_value[0]

    td_error_sq = td_error**2
    td_error_loss = td_error_sum = tf.reduce_sum(td_error_sq)
    loss_argmin = tf.argmin(td_error_sq, 0)
    tf.summary.scalar('loss', td_error_sum)

with tf.name_scope('td_approx'):
    if GRADIENT_CORRECTION:
        [td_approx], td_approx_weights = layer_linear_sum(state_features[0], STATE_FEATURES, 1, True)
        td_approx_sum = tf.reduce_sum((td_error - td_approx)**2)
        tf.summary.scalar('loss', td_approx_sum)
    else:
        td_approx_sum = tf.constant(0.)

opt_conv = tf.train.RMSPropOptimizer(LEARNING_RATE_CONV, momentum=MOMENTUM)
opt_approx = tf.train.RMSPropOptimizer(LEARNING_RATE_LINEAR, momentum=MOMENTUM)
opt_td = tf.train.RMSPropOptimizer(LEARNING_RATE_LINEAR/4, momentum=MOMENTUM)
opt_policy = tf.train.RMSPropOptimizer(LEARNING_RATE_LINEAR/8, momentum=MOMENTUM)
apply_ops= []
if not TEST_ENV:
    # Use auto-differentiation just for convolutional weights
    grad_conv = opt_conv.compute_gradients(td_error_loss, var_list=conv_weights)
    apply_ops += [opt_conv.apply_gradients(grad_conv)]

def reshape_grad_s(x): return tf.reshape(-tf.reduce_sum(x, axis=0), [STATE_FEATURES, 1])
def reshape_grad_sa(x): return tf.reshape(-tf.reduce_sum(x, axis=0), [STATE_FEATURES, ACTION_DIM])

grad_v = td_error*state_features[0]
if GRADIENT_CORRECTION:
    # TD error with gradient correction (TDC)
    grad_v += -GAMMA*td_approx*state_features[1]

    # Learn to approximate the TD error
    grad = (td_error - td_approx)*state_features[0]
    apply_ops += [opt_approx.apply_gradients([(reshape_grad_s(grad), td_approx_weights)])]
apply_ops += [opt_td.apply_gradients([(reshape_grad_s(grad_v), baseline_weights)])]
if COPDAC:
    grad_w = tf.expand_dims(td_error, 2) * state_action_features
    apply_ops += [opt_td.apply_gradients([(reshape_grad_sa(grad_w), w_weights)])]

    grad_theta = tf.expand_dims(state_features[0], 2) * tf.expand_dims(policy_delta, 1)
    apply_ops += [opt_policy.apply_gradients([(reshape_grad_sa(grad_theta), policy_weights)])]

init_vars()

if SAVE_SUMMARIES:
    merged = tf.summary.merge_all()
    apply_ops += [merged]
    train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)

policy_keyboard = True
enable_training = True
def setup_key_actions(a):
    from pyglet.window import key
    def key_press(k, mod):
        global restart, policy_keyboard, enable_training
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
        if k==ord('k'):
            policy_keyboard ^= 1
            print('policy_keyboard:', policy_keyboard)
        if k==ord('t'):
            enable_training ^= 1
            print('enable_training:', enable_training)
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0
    window = env.unwrapped.viewer.window
    window.on_key_press = key_press
    window.on_key_release = key_release

def rl_loop():
    env.reset(); env.render()# Gym seems to need at least 1 reset&render before we get valid observation
    human_action = np.array([0.]*ACTION_DIM)
    setup_key_actions(human_action)

    state = np.zeros([STATE_FRAMES] + FRAME_DIM)
    next_state = np.empty_like(state)
    def trans_state(x):
        dims = len(FRAME_DIM)
        return x.transpose(range(1, dims) + [0, 3]).reshape(STATE_DIM)

    def step_state(num_states):
        if num_states in [0, MAX_EPISODE_STATES]:
            # New episode
            obs = env.reset()
            num_states = 0

        next_reward = 0.
        for frame in range(STATE_FRAMES):
            if SHOW_OBS:
                print('Frame', num_states, frame); imshow(obs)
            env.render()

            # Get obs from env.state instead of env.{reset,step}
            obs = env.unwrapped.state / 255.
            next_state[frame] = obs
            #imshow([obs, test_lcn(obs)])
            obs, reward, done, info = env.step(action)
            next_reward += reward

        return num_states + 1, next_reward

    er_num_saves = 0
    training_its = 0
    def run_training_ops():
        r = sess.run([loss_argmin, td_error_sum, td_approx_sum] + apply_ops)
        save_index=r[0][0]
        save_index = er_num_saves % ER_SIZE
        r = dict(
            #save_index=save_index,
            er_num_saves=er_num_saves,
            td_error_sum=r[1],
            td_approx_sum=r[2],
            policy_action=policy_action)
        print(r)
        #loss_eval = sess.run(td_error); print(loss_eval)
        if SAVE_SUMMARIES:
            summary = r[3]
            train_writer.add_summary(summary, training_its)
        return save_index

    num_states = 0
    while 1:
        action = human_action
        if num_states > 0:
            # Calculate the policy actions:
            [policy_action] = sess.run(policy_ph, feed_dict={state_ph: [trans_state(state)]})
            if not policy_keyboard:
                action = policy_action # On-policy

        do_training = enable_training and er_num_saves >= ER_SIZE
        if do_training:
            run_training_ops()
            training_its += 1

            if not training_its % 5:
                env.render()# Render needed for keyboard events
        else:
            num_states, next_reward = step_state(num_states)
            if num_states >= 2:
                save_index = er_num_saves % ER_SIZE
                sess.run([er_state[save_index].assign(trans_state(state)),
                         er_next_state[save_index].assign(trans_state(next_state)),
                         er_reward[save_index].assign([reward_sum])])
                #imshow([er_state[0][:,:,0:3].eval(), er_state[0][:,:,3:6].eval()])
                er_num_saves += 1

            state, next_state = next_state, state
            reward_sum = next_reward
rl_loop()
