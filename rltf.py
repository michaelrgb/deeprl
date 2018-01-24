import tensorflow as tf, numpy as np
import sys, os, multiprocessing, time
from utils import *
from pprint import pprint

flags = tf.app.flags
# for i in {1..10}; do python rltf.py --async $i & sleep 1; done
flags.DEFINE_integer('async', 0, 'ID of async agent that accumulates gradients on server')
flags.DEFINE_integer('batches_async', 10, 'Batches to wait for from async agents')
flags.DEFINE_integer('batches_keep', 5, 'Batches recorded from user actions')
flags.DEFINE_boolean('replay', False, 'Replay actions recorded in memmap array')
FLAGS = flags.FLAGS

PORT, PROTOCOL = 'localhost:2222', 'grpc'
if not FLAGS.async:
    server = tf.train.Server({'local': [PORT]}, protocol=PROTOCOL, start=True)
sess = tf.InteractiveSession(PROTOCOL+'://'+PORT)

GAMMA = 0.999
STATE_FRAMES = 2
HIDDEN_LAYERS = 2
HIDDEN_NODES = 128

learning_rate_ph = tf.placeholder(tf.float32, ())
opt_td = tf.train.AdamOptimizer(learning_rate_ph, epsilon=0.1)
opt_policy = tf.train.AdamOptimizer(learning_rate_ph/10, epsilon=0.1)

SAVE_SUMMARIES = False

ENV_NAME = os.getenv('ENV')
if not ENV_NAME:
    raise Exception('Missing ENV environment variable')
RESIZE = None
if ENV_NAME == 'CarRacing-v0':
    import gym.envs.box2d
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
ACTION_DISCRETE = not env.action_space.shape
POLICY_SOFTMAX = ACTION_DISCRETE
FRAME_DIM = list(env.observation_space.shape)

CONV_NET = len(FRAME_DIM) == 3
STATE_DIM = FRAME_DIM[:]
if CONV_NET:
    if STATE_DIM[-1] == 3:
        STATE_DIM[-1] = 1
    if RESIZE:
        STATE_DIM[:2] = RESIZE
STATE_DIM[-1] *= STATE_FRAMES

FIRST_BATCH = -FLAGS.batches_keep
LAST_BATCH = FLAGS.batches_async
ER_BATCH_SIZE = 100

training = Struct(enable=True, learning_rate=1e-3, batches_recorded=0, temp_batch=None)
epoch = Struct(count=0, td_error=0)
app = Struct(policy_index=0, quit=False)

def batch_paths(batch_num, path=None):
    if not path:
        path = ENV_NAME + '_' + str(STATE_FRAMES)
    os.system('mkdir -p batches')
    path = 'batches/' + path + '_%i_%s.mmap'
    return [path % (batch_num, 'states'),
            path % (batch_num, 'actions'),
            path % (batch_num, 'rewards')]

REWARDS_GLOBAL = REWARDS_ALL = 1
def mmap_batch(paths, mode):
    batch = Struct(
        states=  np.memmap(paths[0], DTYPE.name, mode, shape=(ER_BATCH_SIZE,) + tuple(STATE_DIM)),
        actions= np.memmap(paths[1], DTYPE.name, mode, shape=(ER_BATCH_SIZE, ACTION_DIM)),
        rewards= np.memmap(paths[2], DTYPE.name, mode, shape=(ER_BATCH_SIZE, REWARDS_GLOBAL)))
    batch.arrays = [batch.states, batch.actions, batch.rewards]
    return batch

if FLAGS.async:
    def init_vars(): pass
    training.current_batch = FLAGS.async-1
else:
    def init_vars(): sess.run(tf.global_variables_initializer())
    training.current_batch = FIRST_BATCH

    if 'w+' in sys.argv:
        # Record new arrays
        app.policy_index = -1
    else:
        training.batches_recorded = FLAGS.batches_keep

    with tf.name_scope('experience_replay'):
        er = Struct(
            states= tf.Variable(tf.zeros([ER_BATCH_SIZE] + STATE_DIM), False, name='states'),
            actions=tf.Variable(tf.zeros([ER_BATCH_SIZE, ACTION_DIM]), False, name='actions'),
            rewards=tf.Variable(tf.zeros([ER_BATCH_SIZE, REWARDS_GLOBAL]), False, name='rewards'))
        er.state =      er.states[:-1]
        er.next_state = er.states[1:]
        er.action =     er.actions[1:]
        er.reward =     er.rewards[1:]
        all_rewards = [er.reward[:, r] for r in range(REWARDS_ALL)] + []

state_ph = tf.placeholder(tf.float32, [None] + STATE_DIM)
action_ph = tf.placeholder(tf.float32, [None, ACTION_DIM])
reward_ph = tf.placeholder(tf.float32, [None, REWARDS_GLOBAL])
frame_ph = tf.placeholder(tf.float32, [None, STATE_FRAMES] + FRAME_DIM)
if CONV_NET:
    frame_to_state = tf.reshape(frame_ph, [-1] + FRAME_DIM) # Move STATE_FRAMES into batches
    if RESIZE:
        frame_to_state = tf.image.resize_images(frame_to_state, RESIZE, tf.image.ResizeMethod.AREA)
    frame_to_state = tf.reduce_mean(frame_to_state/255., axis=-1) # Convert to grayscale
    frame_to_state = tf.reshape(frame_to_state, [-1, STATE_FRAMES] + STATE_DIM[:-1])
    frame_to_state = tf.transpose(frame_to_state, [0, 2, 3, 1]) # Move STATE_FRAMES into channels
else:
    frame_to_state = tf.reshape(frame_ph, [-1] + STATE_DIM)

accum_ops = []
apply_accum_ops = []
zero_accum_ops = []
def accum_value(value):
    accum = tf.Variable(tf.zeros_like(value),trainable=False)
    accum_ops.append(accum.assign_add(value))
    zero_accum_ops.append(accum.assign(tf.zeros_like(value)))
    return accum
def accum_gradient(grads, opt):
    for g,w in grads:
        accum = accum_value(g)
        grad = [(accum, w)]
        apply_accum_ops.append(opt.apply_gradients(grad))

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
all_policy_minmax = []
all_policy_ph = []

def make_acrl():
    states, num_flat_inputs = [frame_to_state], STATE_DIM[0]
    if not FLAGS.async:
        states = [er.state, er.next_state] + states
    activation = tf.nn.softsign if 0 else lambda x: tf.nn.leaky_relu(x, alpha=1e-2)

    def make_conv_net(x):
        conv_channels = STATE_DIM[-1]
        chan_in = conv_channels
        chan_out = 32
        CONV_LAYERS = 4
        for l in range(CONV_LAYERS):
            with tf.name_scope('conv%i' % l):
                x = layer_conv(x, 8, 1, chan_in, chan_out)
                chan_in = chan_out
                chan_out *= 2
                if l < 3: x = max_pool(x, 2, 2)
                x = [activation(i) for i in x]

        # Calculate topmost convolution dimensionality to create fully-connected layer
        init_vars()
        return layer_reshape_flat(x, x[0].eval())

    def make_fully_connected(prev_output, input_size, num_outputs):
        for l in range(HIDDEN_LAYERS):
            with tf.name_scope('fully%i' % l):
                prev_output = layer_fully_connected(prev_output, input_size, HIDDEN_NODES, activation)
                input_size = HIDDEN_NODES
        with tf.name_scope('fully%i' % HIDDEN_LAYERS):
            output = layer_fully_connected(prev_output, input_size, num_outputs, None)
        return output

    with tf.name_scope('policy'):
        policy_layer = states
        if 1:
            if CONV_NET: policy_layer, num_flat_inputs = make_conv_net(states)
        else:
            # Share same conv network as critic
            policy_layer = value_layer
        policy = make_fully_connected(policy_layer, num_flat_inputs, ACTION_DIM)
        policy = [tf.nn.softmax(i) for i in policy] if POLICY_SOFTMAX else policy
        all_policy_ph.append(policy[-1])
        all_policy_minmax.append(tf.concat([
            [tf.reduce_min(policy[0], axis=0)],
            [tf.reduce_max(policy[0], axis=0)]], axis=0))

    if FLAGS.async:
        return

    with tf.name_scope('value'):
        value_layer = states
        if CONV_NET: value_layer, num_flat_inputs = make_conv_net(value_layer)
        if 1:
            value_layer = make_fully_connected(value_layer, num_flat_inputs, 1+ACTION_DIM)
            state_value, next_state_value = value_layer[0][:,0], value_layer[1][:,0]
            policy_one_hot = tf.one_hot(tf.argmax(policy[0], 1), ACTION_DIM)
            off_policy_offset = tf.reduce_sum(er.action - policy_one_hot, -1)
            q_value = state_value + off_policy_offset
        else:
            value_layer = [
                tf.concat([value_layer[0], er.action],axis=1),
                tf.concat([value_layer[0], policy[0]], axis=1),
                tf.concat([value_layer[1], policy[1]], axis=1)]
            q_value, state_value, next_state_value = make_fully_connected(
                value_layer, num_flat_inputs+ACTION_DIM, None)

    # Estimate action advantage from 1-step TD residual
    target_value = all_rewards[r] + GAMMA*next_state_value
    adv_value = target_value - state_value

    if POLICY_SOFTMAX:
        PROB_EPSILON = 1e-2
        prob = tf.reduce_sum(er.action*policy[0], -1)
        adv_value = tf.where(prob < PROB_EPSILON, tf.maximum(0., adv_value), adv_value)
        adv_value = tf.where(prob > 1-PROB_EPSILON, tf.minimum(0., adv_value), adv_value)
        log_prob = tf.log(prob)
    else:
        adv_value = tf.maximum(0., adv_value)
        log_prob = tf.reduce_sum(-(er.action-policy[0])**2, -1)

    repl = gradient_override(log_prob, adv_value)
    grad = opt_policy.compute_gradients(repl, scope_vars('policy'))
    accum_gradient(grad, opt_policy)

    td_error = target_value - q_value
    td_error_sum = tf.reduce_sum(td_error**2)
    all_td_error_sum.append([td_error_sum])

    # TD error with gradient correction (TDC)
    weights = scope_vars('value')
    repl = gradient_override(q_value, td_error)
    grad_s = opt_td.compute_gradients(repl, var_list=weights)
    repl = gradient_override(next_state_value, -GAMMA*td_error)
    grad_s2 = opt_td.compute_gradients(repl, var_list=weights)
    for i in range(len(grad_s)):
        g, g2 = grad_s[i][0], grad_s2[i][0]
        if g2 is not None: g += g2
        grad_s[i] = (g, grad_s[i][1])
    accum_gradient(grad_s, opt_td)

for r in range(REWARDS_ALL):
    with tf.name_scope('ac_' + str(r)):
        make_acrl()

env.reset(); env.render()# Gym needs at least 1 reset&render before valid observation
def setup_key_actions():
    from pyglet.window import key
    a = np.array([0.]*3)

    def key_press(k, mod):
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
        if k==ord('e'):
            app.policy_index = -1
            print('KEYBOARD POLICY')
        if k >= ord('1') and k <= ord('9'):
            app.policy_index = min(REWARDS_ALL-1, int(k - ord('1')))
            print('policy_index:', app.policy_index)
        if k==ord('t'):
            training.enable ^= 1
            print('training.enable:', training.enable)
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0

    window = envu.viewer.window
    window.on_key_press = key_press
    window.on_key_release = key_release
    window.on_close = lambda: setattr(app, 'quit', True)
    return a

state = Struct(frames=np.zeros([STATE_FRAMES] + FRAME_DIM),
               count=0, last_obs=None, done=True)
action = Struct(to_take=None, policy=None, keyboard=setup_key_actions())
def step_to_frames():
    obs = state.last_obs
    env_action = np.argmax(action.to_take) if ACTION_DISCRETE else action.to_take
    reward_sum = 0.
    for frame in range(STATE_FRAMES):
        if state.done:
            # New episode
            env.seed(0) # Same track everytime
            obs = env.reset()
        env.render()
        #imshow([obs, test_lcn(obs, sess)[0]])
        state.frames[frame] = obs

        obs, reward, state.done, info = env.step(env_action)
        if ENV_NAME == 'MountainCar-v0':
            # Mountain car env doesnt give any +reward
            if state.done: reward = 1000.
        reward_sum += reward
    state.last_obs = obs
    return [reward_sum]

ops_single_step = [
    frame_to_state[0],
    all_policy_ph[0][0],# Policy from uploaded state
]
def step_state_upload():
    def action_vector(action_index):
        return [1. if action_index == i else 0. for i in range(ACTION_DIM)]

    a = action.keyboard[:ACTION_DIM]
    if ACTION_DISCRETE:
        # Convert a[0] values to one-hot vector
        a = action_vector(a[0]+1.)

    if state.count > 0:
        if app.policy_index != -1:
            a = action.policy
            if ACTION_DISCRETE:
                # Sample from softmax probabilities
                a = action_vector(np.random.choice(range(ACTION_DIM), p=a))

    if not FLAGS.async and FLAGS.replay:
        a = mmap.actions[state.count]

    # a = [1., 0., 0] if state.frames[0, 1] < 0. else [0., 0., 1.] # MountainCar: go with momentum
    action.to_take = a
    save_rewards = step_to_frames()
    if not FLAGS.async:
        print(dict(state_count=state.count, state='<image>' if CONV_NET else state.frames,
            action_taken=action.to_take, reward=save_rewards))
    if ACTION_DISCRETE:
        a = np.where(a == np.max(action.to_take), 1., 0.)
    ret = sess.run(ops_single_step, feed_dict={frame_ph: [state.frames]})
    frames_to_state = ret[0]
    action.policy = ret[1]

    temp_paths = batch_paths(FLAGS.async, 'temp')
    if not training.temp_batch:
        training.temp_batch = mmap_batch(temp_paths, 'w+')
    batch = training.temp_batch
    batch.states[state.count] = frames_to_state
    batch.rewards[state.count] = save_rewards
    batch.actions[state.count] = a

    state.count += 1
    if state.count == ER_BATCH_SIZE:
        if training.batches_recorded < FLAGS.batches_keep:
            print('Replacing batch #%i' % training.current_batch)
            for a in batch.arrays: a.flush()
            training.temp_batch = None

            # Rename async batch files into server's ER batches.
            save_paths = batch_paths(training.current_batch)
            for dst in save_paths:
                os.system('rm -f ' + dst)
            for src,dst in zip(temp_paths, save_paths):
                os.system('mv ' + src + ' ' + dst)

        if not FLAGS.async:
            training.batches_recorded += 1
            training.current_batch += 1
        state.count = 0

if not FLAGS.async:
    batch_td_errror = tf.concat(all_td_error_sum, axis=0)
    accum_td_error = accum_value(batch_td_errror)
    init_vars()

    ops_batch_upload = [
        er.states.assign(state_ph),
        er.actions.assign(action_ph),
        er.rewards.assign(reward_ph)
    ]

    if SAVE_SUMMARIES:
        merged = tf.summary.merge_all()
        apply_accum_ops += [merged]
        train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)

    accum_ops.append(tf.concat([[p] for p in all_policy_minmax], axis=0))
    accum_ops.append(tf.reduce_sum(er.reward, axis=0))

def train_accum_batch():
    while True:
        try:
            batch = mmap_batch(batch_paths(training.current_batch), 'r')
            feed_dict = {
                state_ph: batch.states,
                action_ph: batch.actions,
                reward_ph: batch.rewards}
        except:
            print('Waiting for batch %i...' % training.current_batch)
            time.sleep(0.1)
            continue
        break

    sess.run(ops_batch_upload, feed_dict=feed_dict)
    # imshow([er.states[10][:,:,i].eval() for i in range(STATE_FRAMES)])

    r = sess.run(accum_ops)
    d = dict(
        batch='%i/%i' % (training.current_batch, LAST_BATCH))
    if app.policy_index != -1:
        d.update(policy_minmax=r[-2][app.policy_index],
                 reward_sum=r[-1][app.policy_index])
    pprint(d)

def train_apply_gradients():
    epoch.prev_td_error = epoch.td_error
    epoch.td_error = sess.run(accum_td_error)[0]
    epoch.count += 1
    training.learning_rate *= 0.5 if epoch.td_error > epoch.prev_td_error else 2**0.5
    training.learning_rate = max(training.learning_rate, 1e-4)
    r = sess.run(apply_accum_ops, feed_dict={learning_rate_ph: training.learning_rate})
    sess.run(zero_accum_ops)
    os.system('clear')
    pprint(dict(
        policy_index=app.policy_index,
        epoch_count=epoch.count,
        epoch_td_error=epoch.td_error,
        learning_rate='%.6f' % training.learning_rate))

    if SAVE_SUMMARIES:
        summary = r[-1]
        train_writer.add_summary(summary, epoch.count)

def rl_loop():
    while not app.quit:
        if FLAGS.async or not training.enable or training.batches_recorded < FLAGS.batches_keep:
            step_state_upload()
            continue

        if training.current_batch < LAST_BATCH:
            train_accum_batch()
            training.current_batch += 1
        else:
            env.render() # Render needed for keyboard events
            train_apply_gradients()
            training.current_batch = FIRST_BATCH
rl_loop()
