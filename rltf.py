GAMMA = 0.999
STATE_FRAMES = 4
ER_SIZE = 500
ER_BATCH = 100
VALUE_LAYERS = 2
POLICY_LAYERS = 2
HIDDEN_NODES = 100
TD_APPROX = False

import tensorflow as tf, numpy as np, sys
from utils import *

LEARNING_RATE = 1e-3
LEARNING_RATE_TD = LEARNING_RATE
if TD_APPROX:
    opt_approx = tf.train.AdamOptimizer(LEARNING_RATE, epsilon=2)
    LEARNING_RATE_TD /= 2
opt_td = tf.train.AdamOptimizer(LEARNING_RATE_TD, epsilon=2)
opt_policy = tf.train.AdamOptimizer(LEARNING_RATE_TD/5, epsilon=2)

SAVE_SUMMARIES = False
MAX_EPISODE_STATES = 10000

ENV_NAME = sys.argv[-1]
RESIZE = None
if ENV_NAME == 'CarRacing-v0':
    car_racing = gym.envs.box2d.car_racing
    car_racing.WINDOW_W = 800 # Default is huge
    car_racing.WINDOW_H = 600
elif ENV_NAME == 'MountainCar-v0':
    pass
elif ENV_NAME == 'FlappyBird-v0':
    import gym_ple
    RESIZE = [128, 128] # Was [512, 288]

import gym
env = gym.make(ENV_NAME)
env._max_episode_steps = None # Disable step limit for now
envu = env.unwrapped

ACTION_DIM = (env.action_space.shape or [env.action_space.n])[0]
FRAME_DIM = list(env.observation_space.shape)

CONV_NET = len(FRAME_DIM) == 3
GREYSCALE = CONV_NET
STATE_DIM = FRAME_DIM[:]
if GREYSCALE and STATE_DIM[-1] == 3:
    STATE_DIM[-1] = 1
STATE_DIM[-1] *= STATE_FRAMES
if RESIZE:
    STATE_DIM[:2] = RESIZE

MMAP_PATH = ENV_NAME + '_' + str(STATE_FRAMES) + '_%s.mmap'
enable_training = False
policy_index = 0
if '-rec' in sys.argv:
    # Record new arrays
    mmap_mode = 'w+'
    policy_index = -1
else:
    # Existing mapped arrays, so train immediately
    mmap_mode = 'r'
    enable_training = True

sess = tf.InteractiveSession()
def init_vars(): sess.run(tf.global_variables_initializer())

REWARDS_GLOBAL = 1
with tf.name_scope('experience_replay'):
    np_frames =     np.memmap(MMAP_PATH % 'frames', DTYPE.name, mmap_mode, shape=(ER_SIZE, STATE_FRAMES) + tuple(FRAME_DIM))
    np_state  =     np.memmap(MMAP_PATH % 'state', DTYPE.name, mmap_mode, shape=(ER_SIZE,) + tuple(STATE_DIM))
    np_action =     np.memmap(MMAP_PATH % 'action', DTYPE.name, mmap_mode, shape=(ER_SIZE, ACTION_DIM))
    np_reward =     np.memmap(MMAP_PATH % 'reward', DTYPE.name, mmap_mode, shape=(ER_SIZE, REWARDS_GLOBAL))
    np_arrays =     [np_frames, np_state, np_action, np_reward]
    er_states =     tf.Variable(tf.zeros([ER_BATCH] + STATE_DIM), False, name='states')
    er_state =      er_states[:-1]
    er_next_state = er_states[1:]
    er_action =     tf.Variable(tf.zeros([ER_BATCH-1, ACTION_DIM]), False, name='action')
    er_reward =     tf.Variable(tf.zeros([ER_BATCH-1, REWARDS_GLOBAL]), False, name='reward')

state_ph = tf.placeholder(tf.float32, [None] + STATE_DIM)
if CONV_NET:
    frame_ph = tf.placeholder(tf.float32, [None, STATE_FRAMES] + FRAME_DIM)
    frame_to_state = tf.reshape(frame_ph, [-1] + FRAME_DIM) # Move STATE_FRAMES into batches
    if RESIZE:
        frame_to_state = tf.image.resize_images(frame_to_state, RESIZE, tf.image.ResizeMethod.AREA)
    frame_to_state = local_contrast_norm(frame_to_state, GAUSS_W)
    frame_to_state = tf.reduce_sum(frame_to_state, axis=-1) # Convert to grayscale
    frame_to_state = tf.reshape(frame_to_state, [-1, STATE_FRAMES] + STATE_DIM[:-1])
    frame_to_state = tf.transpose(frame_to_state, [0, 2, 3, 1]) # Move STATE_FRAMES into channels

def frames_to_state(x):
    if CONV_NET:
        ret = frame_to_state.eval(feed_dict={frame_ph: x})
    else:
        ret = x.reshape([-1] + STATE_DIM)
    return ret
if '-state' in sys.argv:
    # Recalc states from raw frames, in case we change the resize method etc.
    np_state = np.memmap(MMAP_PATH % 'state', DTYPE.name, 'w+', shape=(ER_SIZE,) + tuple(STATE_DIM))
    np_state[:] = frames_to_state(np_frames)
    exit()

er_batch = 0
def upload_batch():
    b = er_batch
    ops = [
        er_states.assign(state_ph),
        er_action.assign(np_action[b:b+ER_BATCH-1]),
        er_reward.assign(np_reward[b:b+ER_BATCH-1])
    ]
    sess.run(ops, feed_dict={state_ph: np_state[b:b+ER_BATCH]})
    # frames = np_frames[b:b+ER_BATCH]; imshow([frames[9][i]/255. for i in range(STATE_FRAMES)])
    # imshow([er_states[9][:,:,i].eval() for i in range(STATE_FRAMES)])
    return

batch_ops = []
apply_accum_ops = []
zero_accum_ops = []
def accum_gradient(grads, opt):
    for g,w in grads:
        accum = tf.Variable(tf.zeros_like(w),trainable=False)
        batch_ops.append(accum.assign_add(g))
        apply_accum_ops.append(opt.apply_gradients([(accum, w)]))
        zero_accum_ops.append(accum.assign(tf.zeros_like(accum)))

# compute_gradients() sums up gradients for all instances, whereas TDC requires a
# multiple of features at s_t+1 to be subtracted from those at s_t.
# Therefore use auto-diff to calculate linear "features" of the Q function,
# and then multiply those features by the TD error etc using custom gradients.
def gradient_override(expr, custom_grad):
    new_op_name = 'new_op_' + str(gradient_override.counter)
    gradient_override.counter += 1
    @tf.RegisterGradient(new_op_name)
    def _grad_(op, grad):
        return -custom_grad
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": new_op_name}):
        return tf.identity(expr)
gradient_override.counter = 0

all_td_error_sum = []
all_td_approx_sum = []
all_policy_minmax = []
all_policy_ph = []
all_rewards = [er_reward[:, r] for r in range(REWARDS_GLOBAL)] + [
]

def make_acrl():
    states, num_flat_inputs = [er_state, er_next_state, state_ph], STATE_DIM[0]
    activation=lambda x: tf.nn.leaky_relu(x, alpha=1e-2)

    def make_conv_net(x):
        conv_channels = STATE_DIM[-1]
        chan_in = conv_channels
        chan_out = 32
        CONV_LAYERS = 3
        for l in range(CONV_LAYERS):
            with tf.name_scope('layer%i' % l):
                x = layer_conv(x, 9, 1, chan_in, chan_out)
                chan_in = chan_out; chan_out *= 2
                if l != CONV_LAYERS-1:
                    x = max_pool(x, 3, 3)
                x = [activation(i) for i in x]

        # Calculate topmost convolution dimensionality to create fully-connected layer
        init_vars()
        return layer_reshape_flat(x, x[0].eval())

    def make_state_network(prev_output, num_layers, num_outputs, last_layer_activation=None):
        input_size = num_flat_inputs
        for l in range(num_layers):
            prev_output = layer_fully_connected(prev_output, input_size, HIDDEN_NODES, activation)
            input_size = HIDDEN_NODES
        output = layer_fully_connected(prev_output, input_size, num_outputs, last_layer_activation)
        return output

    with tf.name_scope('state_value'):
        value_layer = states
        if CONV_NET: value_layer, num_flat_inputs = make_conv_net(value_layer)
        # Baseline value, independent of action a
        state_value = make_state_network(value_layer, VALUE_LAYERS, None)

    with tf.name_scope('policy'):
        if 1:
            policy_layer = states
            if CONV_NET: policy_layer, _ = make_conv_net(policy_layer)
        else:
            # Share same conv network as critic
            policy_layer = value_layer
        policy = make_state_network(policy_layer, POLICY_LAYERS, ACTION_DIM)
        all_policy_ph.append(policy[2])
        all_policy_minmax.append([tf.reduce_min(policy[0], axis=0), tf.reduce_max(policy[0], axis=0)])

    def make_advantage_network(offset):
        offset_dim = int(offset.shape[1])
        layer = make_state_network(value_layer, VALUE_LAYERS, offset_dim)
        adv_action = [i * offset for i in layer]
        adv_value = [tf.reduce_sum(i, 1) for i in adv_action]
        return adv_action, adv_value

    with tf.name_scope('action_advantage'):
        # Advantage is linear in (action-policy)
        off_policy = er_action-policy[0]
        adv_action, adv_value = make_advantage_network(off_policy)

        adv_grad = tf.gradients(adv_value[0], off_policy)[0]
        policy_grad = adv_grad * adv_action[0]

    if TD_APPROX:
        with tf.name_scope('td_approx'):
            approx = make_state_network(value_layer, VALUE_LAYERS, None)
            td_approx = approx[0]

            _, approx = make_advantage_network(
                tf.concat([tf.maximum(off_policy, 0.), tf.maximum(-off_policy, 0.)], axis=1))
            td_approx += approx[0]

    with tf.name_scope('q_value'):
        q_value = [state_value[0]+adv_value[0], state_value[1]]

    with tf.name_scope('td_error'):
        td_error = all_rewards[r] + GAMMA*q_value[1] - q_value[0]

        td_error_sq = td_error**2
        td_error_sum = tf.reduce_sum(td_error_sq)
        all_td_error_sum.append([td_error_sum])
        #loss_argmin = tf.argmin(td_error_sq, 0)

    if TD_APPROX:
        td_approx_sum = tf.reduce_sum((td_error - td_approx)**2)
        tf.summary.scalar('loss', td_approx_sum)
        all_td_approx_sum.append([td_approx_sum])

        # Approximate the TD error
        repl = gradient_override(td_approx, td_error-td_approx)
        grad = opt_approx.compute_gradients(repl, var_list=scope_vars('td_approx'))
        accum_gradient(grad, opt_approx)

    repl = gradient_override(policy[0], policy_grad)
    grad = opt_policy.compute_gradients(repl, var_list=scope_vars('policy'))
    accum_gradient(grad, opt_policy)

    # TD error with gradient correction (TDC)
    for (value, weights) in [(q_value,
        scope_vars('state_value')+scope_vars('action_advantage'))]:
        repl = gradient_override(value[0], td_error)
        grad_s = opt_td.compute_gradients(repl, var_list=weights)
        repl = gradient_override(value[1], -GAMMA*(td_approx if TD_APPROX else td_error))
        grad_s2 = opt_td.compute_gradients(repl, var_list=weights)
        for i in range(len(grad_s)):
            g, g2 = grad_s[i][0], grad_s2[i][0]
            if g2 is not None: g += g2
            grad_s[i] = (g, grad_s[i][1])
        accum_gradient(grad_s, opt_td)

for r in range(len(all_rewards)):
    with tf.name_scope('ac_' + str(r)):
        make_acrl()

td_error_sum = tf.concat(all_td_error_sum, axis=0)
td_approx_sum = tf.concat(all_td_approx_sum, axis=0) if TD_APPROX else tf.constant(0.)

if SAVE_SUMMARIES:
    merged = tf.summary.merge_all()
    batch_ops += [merged]
    train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)

init_vars()
def setup_key_actions():
    from pyglet.window import key
    a = np.array([0.]*3)
    def key_press(k, mod):
        global restart, policy_index, enable_training
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
        if k==ord('e'):
            policy_index = -1
            print('KEYBOARD POLICY')
        if k >= ord('1') and k <= ord('9'):
            policy_index = min(len(all_rewards)-1, int(k - ord('1')))
            print('policy_index:', policy_index)
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
    return a

def rl_loop():
    env.reset(); env.render()# Gym seems to need at least 1 reset&render before we get valid observation
    keyboard_action = setup_key_actions()

    er_num_saves = 0
    state = np.ones([1, STATE_FRAMES] + FRAME_DIM)
    next_state = np.ones_like(state)

    def step_state(states_sofar, obs):
        env_reset = states_sofar in [0, MAX_EPISODE_STATES]
        env_action = action if env.action_space.shape else np.argmax(action)
        reward_sum = 0.
        for frame in range(STATE_FRAMES):
            if env_reset:
                # New episode
                env.seed(0) # Same track everytime
                obs = env.reset()
                states_sofar = 0
            env.render()
            #imshow([obs, test_lcn(obs, sess)])
            next_state[0, frame] = obs

            obs, reward, env_reset, info = env.step(env_action)
            if ENV_NAME == 'MountainCar-v0':
                # Mountain car env doesnt give any +reward
                if env_reset: reward = 1000.
            reward_sum += reward
        print(dict(er_num_saves=er_num_saves, state='<image>' if CONV_NET else state,
            action_taken=action, reward=reward_sum))
        return states_sofar + 1, obs, reward_sum

    training_epochs = 0
    def run_training_ops():
        r = sess.run([td_error_sum, td_approx_sum] + batch_ops)
        #save_index=r[0][0]
        save_index = er_num_saves % ER_SIZE
        if SAVE_SUMMARIES:
            summary = r[-1]
            train_writer.add_summary(summary, training_epochs)
        r = dict(
            #policy_weights='\n'.join(str(w.eval()) for w in policy_weights),
            training_epochs=training_epochs,
            er_batch='%i/%i' % (er_batch, ER_SIZE),
            td_error_sum=r[0],
            td_approx_sum=r[1],
            policy_index=policy_index,
            policy_action=policy_action,
            policy_minmax=sess.run(all_policy_minmax[policy_index]))
        for k in sorted(r.keys()):
            print(k + ': ' + str(r[k]))
        print(' ')
        return save_index

    global policy_index, enable_training, er_batch
    states_sofar = 0
    last_obs = None
    while 1:
        action = keyboard_action
        if not env.action_space.shape:
            # Convert a[0] values to one-hot vector
            action = [1. if action[0]+1. == i else 0. for i in range(ACTION_DIM)]

        policy_action = None
        if states_sofar > 0:
            # Calculate the policy actions:
            if policy_index != -1:
                [policy_action] = sess.run(all_policy_ph[policy_index], feed_dict={state_ph: frames_to_state(state)})
                action = policy_action

        do_env_step = True
        if enable_training:
            upload_batch()
            run_training_ops()
            training_epochs += 1

            er_batch += ER_BATCH
            if er_batch == ER_SIZE:
                er_batch = 0
                sess.run(apply_accum_ops)
                sess.run(zero_accum_ops)

            # The GL env slows down training
            #if training_epochs % 10:
            do_env_step = False

        if do_env_step:
            # action = [1., 0., 0] if next_state[0, 1] < 0. else [0., 0., 1.] # MountainCar: go with momentum
            states_sofar, last_obs, reward_sum = step_state(states_sofar, last_obs)
            if states_sofar >= 2 and np_frames.flags.writeable:#er_num_saves < ER_SIZE:
                save_index = er_num_saves % ER_SIZE
                if policy_index != -1:
                    # Keep some demonstrated actions
                    save_index = ER_SIZE/2 + er_num_saves % int(ER_SIZE/2)
                np_frames[save_index] = state
                np_state[save_index] = frames_to_state(state)[0]
                np_action[save_index] = action
                np_reward[save_index] = [reward_sum]

                er_num_saves += 1
                if er_num_saves == ER_SIZE:
                    for array in np_arrays:
                        array.flush()
                    enable_training = True
                    # Switch to target policy
                    policy_index = 0

            state, next_state = next_state, state
        else:
            # Render needed for keyboard events
            if not training_epochs % 5: env.render()

rl_loop()
