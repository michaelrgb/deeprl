# coding=utf8

GAMMA = 0.99
STATE_FRAMES = 4
ER_SIZE = 1000
GRADIENT_CORRECTION = True
COPDAC = True
STATE_FEATURES = 1000
FULLY_CONNECTED_LAYERS = 3
LEARNING_RATE_LINEAR = 1e-5
LEARNING_RATE_CONV = 1e-12
MOMENTUM = 0.9

SAVE_SUMMARIES = False
SHOW_OBS = False
MAX_EPISODE_STATES = 10000

import tensorflow as tf, numpy as np
from utils import *

import gym
ENV_NAME = '''
CarRacing-v0
MountainCar-v0
'''.splitlines()[-1]
env = gym.make(ENV_NAME)
envu = env.unwrapped
if ENV_NAME == 'CarRacing-v0':
    car_racing = gym.envs.box2d.car_racing
    car_racing.WINDOW_W = 800 # Default is huge
    car_racing.WINDOW_H = 600
elif ENV_NAME == 'MountainCar-v0':
    envu.max_speed *= 2 # Cant seem to get up mountain
    STATE_FRAMES = 1
FRAME_DIM = list(env.observation_space.shape)
ACTION_DIM = (env.action_space.shape or [env.action_space.n])[0]

CONV_NET = len(FRAME_DIM) == 3
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

def make_conv(x):
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
    x = [lrelu(i) for i in x]
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
with tf.name_scope('conv_net'):
    states = [er_state, er_next_state, state_ph]
    if CONV_NET:
        conv_net, shared_weights = make_conv(states)
        # Calculate topmost convolution dimensionality to create fully-connected layer
        init_vars()
        conv_net, num_flat_inputs = layer_reshape_flat(conv_net, conv_net[0].eval())
    else:
        shared_weights = []
        conv_net, num_flat_inputs = states, STATE_DIM[0]
with tf.name_scope('state_features'):
    state_features = conv_net
    for i in range(FULLY_CONNECTED_LAYERS):
        state_features, w = layer_fully_connected(state_features, num_flat_inputs, STATE_FEATURES)
        shared_weights += w
        num_flat_inputs = STATE_FEATURES

if COPDAC:
    with tf.name_scope('policy'):
        # a = μ_θ(s)
        [policy, policy_ph], policy_weights = layer_linear_sum([state_features[0], state_features[2]], STATE_FEATURES, ACTION_DIM)
        [policy_grad], w_weights = layer_linear_sum(state_features[0], STATE_FEATURES, ACTION_DIM)

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
    td_error_sum = tf.reduce_sum(td_error_sq)
    loss_argmin = tf.argmin(td_error_sq, 0)
    tf.summary.scalar('loss', td_error_sum)

with tf.name_scope('td_approx'):
    if GRADIENT_CORRECTION:
        [td_approx], td_approx_weights = layer_linear_sum(state_features[0], STATE_FEATURES, 1, True)
        td_approx_sum = tf.reduce_sum((td_error - td_approx)**2)
        tf.summary.scalar('loss', td_approx_sum)
    else:
        td_approx_sum = tf.constant(0.)

opt_approx = tf.train.RMSPropOptimizer(LEARNING_RATE_LINEAR, momentum=MOMENTUM)
opt_td = tf.train.RMSPropOptimizer(LEARNING_RATE_LINEAR/4, momentum=MOMENTUM)
opt_policy = tf.train.RMSPropOptimizer(LEARNING_RATE_LINEAR/100, momentum=MOMENTUM)
apply_ops = []

# compute_gradients() sums up gradients for all instances, whereas TDC requires a
# multiple of features at s_t+1 to be subtracted from those at s_t.
# Therefore use auto-diff to calculate linear "features" of the Q function,
# and then multiply those features by the TD error etc using custom gradients.
def replace_gradient(expr, custom_grad):
    new_op_name = 'new_op_' + str(replace_gradient.counter)
    replace_gradient.counter += 1
    @tf.RegisterGradient(new_op_name)
    def _grad_(op, grad):
        return -custom_grad
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": new_op_name}):
        return tf.identity(expr)
replace_gradient.counter = 0

repl = replace_gradient(q_value[0], td_error)
td_weights = shared_weights + [baseline_weights]
grad_s = opt_td.compute_gradients(repl, var_list=td_weights)
if COPDAC:
    grad_w = opt_td.compute_gradients(repl, var_list=shared_weights+[w_weights])
    apply_ops += [opt_td.apply_gradients(grad_w)]

if GRADIENT_CORRECTION:
    # TD error with gradient correction (TDC)
    repl = replace_gradient(q_value[1], -GAMMA*td_approx)
    grad_s2 = opt_td.compute_gradients(repl, var_list=td_weights)
    for i in range(len(td_weights)):
        grad_s[i] = (grad_s[i][0] + grad_s2[i][0], grad_s[i][1])
apply_ops += [opt_td.apply_gradients(grad_s)]

# Approximate the TD error
repl = replace_gradient(td_approx, td_error-td_approx)
grad = opt_approx.compute_gradients(repl, var_list=shared_weights+[td_approx_weights])
apply_ops += [opt_approx.apply_gradients(grad)]

if COPDAC:
    repl = replace_gradient(policy, policy_grad)
    grad = opt_policy.compute_gradients(repl, var_list=policy_weights)
    apply_ops += [opt_policy.apply_gradients(grad)]

init_vars()

if SAVE_SUMMARIES:
    merged = tf.summary.merge_all()
    apply_ops += [merged]
    train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)

on_policy = False
enable_training = True
def setup_key_actions(a):
    from pyglet.window import key
    def key_press(k, mod):
        global restart, on_policy, enable_training
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
        if k==ord('p'):
            on_policy ^= 1
            print('on_policy:', on_policy)
        if k==ord('t'):
            enable_training ^= 1
            print('enable_training:', enable_training)
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0
    window = envu.viewer.window
    window.on_key_press = key_press
    window.on_key_release = key_release

ACTIONS_CSV = 'actions_' + ENV_NAME + '.csv'
try: actions_csv = np.loadtxt(open(ACTIONS_CSV, 'rb'), delimiter=',', skiprows=0)
except: actions_csv = []

def rl_loop():
    env.reset(); env.render()# Gym seems to need at least 1 reset&render before we get valid observation
    keyboard_action = np.array([0.]*ACTION_DIM)
    setup_key_actions(keyboard_action)

    er_num_saves = 0
    state = np.zeros([STATE_FRAMES] + FRAME_DIM)
    next_state = np.empty_like(state)
    def trans_state(x):
        if CONV_NET:
            dims = len(FRAME_DIM)
            x = x.transpose(range(1, dims) + [0, 3]) / 255.
        return x.reshape(STATE_DIM)

    def step_state(states_sofar):
        if states_sofar in [0, MAX_EPISODE_STATES]:
            # New episode
            obs = env.reset()
            states_sofar = 0

        env_action = action if env.action_space.shape else np.argmax(action)
        reward_sum = 0.
        for frame in range(STATE_FRAMES):
            if SHOW_OBS:
                print('Frame', states_sofar, frame); imshow(obs)
            env.render()

            obs, reward, done, info = env.step(env_action)
            next_state[frame] = obs
            #imshow([obs, test_lcn(obs)])
            if ENV_NAME == 'MountainCar-v0':
                done = False
                # Mountain car env doesnt give any +reward
                if obs[0] >= envu.goal_position:
                    reward = 1000.
                    done = True
            reward_sum += reward
        print('ER saves:', er_num_saves, 'Action taken:', action, 'Reward:', reward_sum)
        return states_sofar + 1, reward_sum, done

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

    global on_policy
    states_sofar = 0
    while 1:
        action = keyboard_action
        if not env.action_space.shape:
            # Convert a[0] values to one-hot vector
            action = [1. if action[0]+1. == i else 0. for i in range(ACTION_DIM)]

        if states_sofar > 0:
            # Calculate the policy actions:
            [policy_action] = sess.run(policy_ph, feed_dict={state_ph: [trans_state(state)]})
            if on_policy:
                action = policy_action # On-policy

        do_env_step = True
        do_training = enable_training and er_num_saves >= ER_SIZE
        if do_training:
            run_training_ops()
            training_its += 1
            # The GL env slows down training
            #if training_its % 10:
            do_env_step = False

        if do_env_step:
            if er_num_saves < len(actions_csv):
                action = actions_csv[er_num_saves]

            states_sofar, reward_sum, done = step_state(states_sofar)
            if states_sofar >= 2:# and er_num_saves < ER_SIZE:
                save_index = er_num_saves % ER_SIZE
                if on_policy:
                    # Keep all the demonstrated actions
                    save_index = ER_SIZE/2 + er_num_saves % int(ER_SIZE/2)
                sess.run([er_state[save_index].assign(trans_state(state)),
                          er_next_state[save_index].assign(trans_state(next_state)),
                          er_action[save_index].assign(action),
                          er_reward[save_index].assign([reward_sum])])
                #imshow([er_state[0][:,:,0:3].eval(), er_state[0][:,:,3:6].eval()])
                er_num_saves += 1

                if er_num_saves == ER_SIZE/2:
                    np.savetxt(ACTIONS_CSV, er_action[0:er_num_saves].eval(), delimiter=',')
                    # Switch to target policy
                    on_policy = True

            state, next_state = next_state, state
            if done:
                states_sofar = 0
        else:
            # Render needed for keyboard events
            if not training_its % 5: env.render()

rl_loop()
