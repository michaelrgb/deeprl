import tensorflow as tf, numpy as np
from tensorflow.contrib import rnn
import sys, os, multiprocessing, time
from utils import *
from pprint import pprint

flags = tf.app.flags
# for i in {1..10}; do python rltf.py --async $i & sleep 1; done
flags.DEFINE_integer('async', 0, 'ID of async agent that accumulates gradients on server')
flags.DEFINE_integer('batch_async', 10, 'Batches to wait for from async agents')
flags.DEFINE_integer('batch_per_async', 1, 'Batches recorded per async agent')
flags.DEFINE_integer('batch_keep', 5, 'Batches recorded from user actions')
flags.DEFINE_boolean('replay', False, 'Replay actions recorded in memmap array')
flags.DEFINE_boolean('record', False, 'Record over kept batches')
flags.DEFINE_boolean('er_all', False, 'Keep all batches in ER memory')
flags.DEFINE_float('learning_rate', 1e-3, 'Minimum learning rate')
FLAGS = flags.FLAGS

PORT, PROTOCOL = 'localhost:2222', 'grpc'
if not FLAGS.async:
    server = tf.train.Server({'local': [PORT]}, protocol=PROTOCOL, start=True)
sess = tf.InteractiveSession(PROTOCOL+'://'+PORT)

GAMMA = 0.999
STATE_FRAMES = 2
HIDDEN_LAYERS = 2
HIDDEN_NODES = 128
LSTM_NODES = 256
LSTM_NET = True

SAVE_SUMMARIES = False

ENV_NAME = os.getenv('ENV')
if not ENV_NAME:
    raise Exception('Missing ENV environment variable')
if ENV_NAME == 'CarRacing-v0':
    import gym.envs.box2d
    car_racing = gym.envs.box2d.car_racing
    car_racing.WINDOW_W = 800 # Default is huge
    car_racing.WINDOW_H = 600
elif ENV_NAME == 'FlappyBird-v0':
    import gym_ple # [512, 288]
elif 'Bullet' in ENV_NAME:
    import pybulletgym.envs

import gym
env = gym.make(ENV_NAME)
#env._max_episode_steps = None # Disable step limit
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
    FRAME_LCN = False
    RESIZE = [84, 84] # DQN
    if RESIZE:
        STATE_DIM[:2] = RESIZE
STATE_DIM[-1] *= STATE_FRAMES

FIRST_BATCH = -FLAGS.batch_keep
LAST_BATCH = FLAGS.batch_async
ER_BATCH_SIZE = 100
ER_BATCHES = FLAGS.batch_keep + FLAGS.batch_async if FLAGS.er_all else 1

training = Struct(enable=True, learning_rate=FLAGS.learning_rate, batches_recorded=0, batches_mtime={}, temp_batch=None)
epoch = Struct(count=0, td_error=0, prev_td_error=0)
app = Struct(policy_index=0, sample_actions=True, quit=False)

def batch_paths(batch_num, path=None):
    if not path:
        path = ENV_NAME + '_' + str(STATE_FRAMES)
    os.system('mkdir -p batches')
    path = 'batches/' + path + '_%i_%s.mmap'
    return [path % (batch_num, 'states'),
            path % (batch_num, 'actions'),
            path % (batch_num, 'rewards')]

REWARDS_GLOBAL = REWARDS_ALL = 1
def mmap_batch(paths, mode, only_actions=False):
    batch = Struct(actions=np.memmap(paths[1], DTYPE.name, mode, shape=(ER_BATCH_SIZE, ACTION_DIM)))
    if only_actions:
        return batch.actions
    batch.states = np.memmap(paths[0], DTYPE.name, mode, shape=(ER_BATCH_SIZE,) + tuple(STATE_DIM))
    batch.rewards = np.memmap(paths[2], DTYPE.name, mode, shape=(ER_BATCH_SIZE, REWARDS_GLOBAL))
    batch.arrays = [batch.states, batch.actions, batch.rewards]
    return batch

ph = Struct(learning_rate=tf.placeholder(tf.float32, ()),
    enable_tdc=tf.placeholder(tf.bool, ()),
    batch_idx=tf.placeholder(tf.int32, ()),
    state=tf.placeholder(tf.float32, [None] + STATE_DIM),
    action=tf.placeholder(tf.float32, [None, ACTION_DIM]),
    reward=tf.placeholder(tf.float32, [None, REWARDS_GLOBAL]),
    frame=tf.placeholder(tf.float32, [None, STATE_FRAMES] + FRAME_DIM))
opt_td = tf.train.AdamOptimizer(ph.learning_rate, epsilon=0.1)
opt_policy = tf.train.AdamOptimizer(ph.learning_rate/10, epsilon=0.1)

if FLAGS.async:
    FIRST_BATCH = (FLAGS.async-1)*FLAGS.batch_per_async
    def init_vars(): pass
else:
    def init_vars(): sess.run(tf.global_variables_initializer())

    if FLAGS.record or FLAGS.replay:
        # Record new arrays
        app.policy_index = -1
    else:
        training.batches_recorded = FLAGS.batch_keep

    with tf.name_scope('experience_replay'):
        er = Struct(
            states= tf.Variable(tf.zeros([ER_BATCHES, ER_BATCH_SIZE] + STATE_DIM), False, name='states'),
            actions=tf.Variable(tf.zeros([ER_BATCHES, ER_BATCH_SIZE, ACTION_DIM]), False, name='actions'),
            rewards=tf.Variable(tf.zeros([ER_BATCHES, ER_BATCH_SIZE, REWARDS_GLOBAL]), False, name='rewards'))
        er.state =      er.states[ph.batch_idx][:-1]
        er.next_state = er.states[ph.batch_idx][1:]
        er.action =     er.actions[ph.batch_idx][1:]
        er.reward =     er.rewards[ph.batch_idx][1:]
        all_rewards = [er.reward[:, r] for r in range(REWARDS_ALL)] + []
training.current_batch = FIRST_BATCH

if CONV_NET:
    frame_to_state = tf.reshape(ph.frame, [-1] + FRAME_DIM) # Move STATE_FRAMES into batches
    if RESIZE:
        frame_to_state = tf.image.resize_images(frame_to_state, RESIZE, tf.image.ResizeMethod.AREA)
    if FRAME_LCN:
        frame_to_state = local_contrast_norm(frame_to_state, GAUSS_W)
        frame_to_state = tf.reduce_sum(frame_to_state, axis=-1)
    else:
        frame_to_state = tf.reduce_mean(frame_to_state/255., axis=-1) # Convert to grayscale
    frame_to_state = tf.reshape(frame_to_state, [-1, STATE_FRAMES] + STATE_DIM[:-1])
    frame_to_state = tf.transpose(frame_to_state, [0, 2, 3, 1]) # Move STATE_FRAMES into channels
else:
    frame_to_state = tf.reshape(ph.frame, [-1] + STATE_DIM)

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
# Therefore use custom gradients to pre-multiply instances before batch is summed together.
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
all_policy = []
stats = Struct(reward_sum=[], on_policy_count=[], max_advantage=[], _policy_minmax=[])

activation = tf.nn.leaky_relu if 1 else tf.nn.softsign
def make_fully_connected(x, outputs):
    for l in range(HIDDEN_LAYERS):
        with tf.name_scope('fully%i' % l):
            x = layer_fully_connected(x, HIDDEN_NODES, activation)
    with tf.name_scope('fully%i' % HIDDEN_LAYERS):
        output = layer_fully_connected(x, outputs, None)
    return output

def make_conv_net(x):
    if not CONV_NET:
        # If no conv, add layers before LSTM
        for l in range(HIDDEN_LAYERS):
            with tf.name_scope('pre%i' % l):
                x = layer_fully_connected(x, HIDDEN_NODES, activation)
                x = [tf.nn.local_response_normalization(
                    tf.expand_dims(tf.expand_dims(i, 1), 1))[:,0,0,:] for i in x]
        return x

    chan_in = STATE_DIM[-1]
    chan_out = 32
    CONV_LAYERS = 3
    for l in range(CONV_LAYERS):
        with tf.name_scope('conv%i' % l):
            x = layer_conv(x, 7, 1, chan_in, chan_out)
            chan_in = chan_out
            chan_out *= 2
            if l < 3: x = max_pool(x, 4, 2)
            x = [activation(i) for i in x]
            x = [tf.nn.local_response_normalization(i) for i in x]
    return layer_reshape_flat(x, x[0].shape)[0]

def make_acrl():
    with tf.name_scope('policy'):
        policy_layer = make_conv_net([frame_to_state] if FLAGS.async else [er.state, frame_to_state])
        p = Struct()
        if LSTM_NET:
            cell = rnn.BasicLSTMCell(LSTM_NODES)
            scope = tf.contrib.framework.get_name_scope()
            if not FLAGS.async:
                [policy_layer[0]], _ = rnn.static_rnn(cell, [policy_layer[0]], dtype=tf.float32, scope=scope)
            p.policy_lstm_saved = tf.contrib.rnn.LSTMStateTuple(
                *[tf.Variable(s, trainable=False) for s in cell.zero_state(1, DTYPE)])
            [policy_layer[-1]], final_state = rnn.static_rnn(cell, [policy_layer[-1]], initial_state=p.policy_lstm_saved, scope=scope)
            p.lstm_save_ops = [p.policy_lstm_saved[i].assign(final_state[i]) for i in range(2)]

        policy = make_fully_connected(policy_layer, ACTION_DIM)
        policy = [tf.nn.softmax(i) for i in policy] if POLICY_SOFTMAX else policy
        p.policy = policy[-1]
        all_policy.append(p)
        stats._policy_minmax.append(tf.concat([
            [tf.reduce_min(policy[0], axis=0)],
            [tf.reduce_max(policy[0], axis=0)]], axis=0))

    if FLAGS.async:
        return

    with tf.name_scope('value'):
        value_layer = make_conv_net([er.state, er.next_state, frame_to_state])
        if LSTM_NET:
            cell = rnn.BasicLSTMCell(LSTM_NODES)
            scope = tf.contrib.framework.get_name_scope()
            value_layer, _ = rnn.static_rnn(cell, value_layer, dtype=tf.float32, scope=scope)

        if 0:
            value_layer = make_fully_connected(value_layer, 1+ACTION_DIM)
            [state_value, next_state_value, _] = [i[:,0] for i in value_layer]
            adv_direction = value_layer[0][:,1:]
        else:
            [state_value, next_state_value, _] = make_fully_connected(value_layer, None)
            [adv_direction] = make_fully_connected(value_layer[0], ACTION_DIM)

        # Learn from a minimum off_policy to keep advantage function stable when off_policy increases.
        off_policy = er.action - policy[0]
        off_policy = tf.where(tf.abs(off_policy)<1e-2, tf.zeros(off_policy.shape), off_policy)
        adv_action = adv_direction*off_policy
        q_value = state_value + tf.reduce_sum(adv_action, -1)

    policy_action = tf.one_hot(tf.argmax(policy[0], 1), ACTION_DIM) if ACTION_DISCRETE else policy[0]
    is_on_policy = tf.reduce_sum(tf.abs(er.action-policy_action), -1) < 0.1
    def multi_step(max_recurse, i=0):
        z = tf.zeros(i)
        next = tf.concat([next_state_value[i:], z], 0)
        return tf.concat([all_rewards[r][i:], z], 0) + GAMMA*(tf.where(
            tf.concat([is_on_policy[i+1:], tf.zeros(i+1, dtype=tf.bool)], 0),
            multi_step(max_recurse, i+1),
            next) if i < max_recurse else next)
    target_value = multi_step(10)# 0 is 1-step TD
    td_error = target_value - q_value
    all_td_error_sum.append([tf.reduce_sum(td_error**2)])

    if POLICY_SOFTMAX:
        prob = tf.reduce_sum(policy[0]*er.action, -1)
        log_prob = tf.log(tf.clip_by_value(prob, 1e-20, 1-1e-20))
    else:
        log_prob = -tf.reduce_sum((policy[0]-er.action)**2, -1)
        prob = tf.exp(log_prob)

    adv = target_value - state_value
    adv = tf.where(prob < 1e-2, tf.maximum(0., adv),
          tf.where(prob > 1-1e-2, tf.minimum(0., adv), adv))

    repl = gradient_override(log_prob, adv)
    grad = opt_policy.compute_gradients(repl, scope_vars('policy'))
    accum_gradient(grad, opt_policy)

    stats.max_advantage.append(tf.reduce_max(tf.abs(adv)))
    stats.on_policy_count.append(tf.reduce_sum(tf.cast(is_on_policy, tf.int32)))
    stats.reward_sum.append(tf.reduce_sum(all_rewards[r]))

    # TD error with gradient correction (TDC)
    weights = scope_vars('value')
    repl = gradient_override(q_value, td_error)
    grad_s = opt_td.compute_gradients(repl, weights)
    repl = gradient_override(next_state_value, -GAMMA*td_error)
    grad_s2 = opt_td.compute_gradients(repl, weights)
    for i in range(len(grad_s)):
        (g, w), g2 = grad_s[i], grad_s2[i][0]
        if g2 is not None: g += tf.where(ph.enable_tdc, g2, tf.zeros(g2.shape))
        grad_s[i] = (g, w)
    accum_gradient(grad_s, opt_td)

for r in range(REWARDS_ALL):
    with tf.name_scope('ac_' + str(r)):
        make_acrl()

envu.isRender = True # pybullet-gym
env.reset(); env.render()# Gym needs at least 1 reset&render before valid observation
def setup_key_actions():
    from pyglet.window import key
    a = np.array([0.]*max(3, ACTION_DIM))

    def key_press(k, mod):
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
        if k==ord('e'):
            app.policy_index = -1
            print('KEYBOARD POLICY')
        elif k >= ord('1') and k <= ord('9'):
            app.policy_index = min(REWARDS_ALL-1, int(k - ord('1')))
        elif k==ord('s'):
            app.sample_actions ^= True
        elif k==ord('t'):
            training.enable ^= True
        else:
            return
        print(dict(
            policy_index=app.policy_index,
            training_enable=training.enable,
            sample_actions=app.sample_actions))

    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0

    if not hasattr(envu, 'viewer'): # pybullet-gym
        return a
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
            reward = 1. if state.done else 0.
        reward_sum += reward
    state.last_obs = obs
    return [reward_sum]

ops_single_step = [
    frame_to_state[0],
    all_policy[0].policy[0],# Policy from uploaded state,
] + ([op for p in all_policy for op in p.lstm_save_ops] if LSTM_NET else [])
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
            if app.sample_actions:
                if POLICY_SOFTMAX:
                    # Sample from softmax probabilities
                    a = action_vector(np.random.choice(range(ACTION_DIM), p=a))
                else:
                    rand = 0.2
                    a += rand*(2*np.random.random(ACTION_DIM)-1.)
                    if not ACTION_DISCRETE: a = np.clip(a, -1., 1.)

    save_paths = batch_paths(training.current_batch)
    if not FLAGS.async and FLAGS.replay:
        a = mmap_batch(save_paths, 'r', only_actions=True)[state.count]

    # a = [1., 0., 0] if state.frames[0, 1] < 0. else [0., 0., 1.] # MountainCar: go with momentum
    action.to_take = a
    save_rewards = step_to_frames()
    if 0:#not FLAGS.async:
        print(dict(state_count=state.count, state='<image>' if CONV_NET else state.frames,
            action_taken=action.to_take, reward=save_rewards))
    if ACTION_DISCRETE:
        a = np.where(a == np.max(action.to_take), 1., 0.)
    ret = sess.run(ops_single_step, feed_dict={ph.frame: [state.frames]})
    frames_to_state = ret[0]
    action.policy = ret[1]

    temp_paths = batch_paths(FLAGS.async, 'temp')
    if not training.temp_batch:
        training.temp_batch = mmap_batch(temp_paths, 'w+')
    batch = training.temp_batch
    batch.states[state.count] = frames_to_state
    # imshow([frames_to_state[:,:,i] for i in range(STATE_FRAMES)])
    batch.rewards[state.count] = save_rewards
    batch.actions[state.count] = a

    state.count += 1
    if state.count == ER_BATCH_SIZE:
        if FLAGS.async or training.batches_recorded < FLAGS.batch_keep:
            print('Replacing batch #%i' % training.current_batch)
            for a in batch.arrays: a.flush()
            training.temp_batch = None

            # Rename async batch files into server's ER batches.
            for dst in save_paths:
                os.system('rm -f ' + dst)
            for src,dst in zip(temp_paths, save_paths):
                os.system('mv ' + src + ' ' + dst)

        training.batches_recorded += 1
        training.current_batch += 1
        if FLAGS.async and training.batches_recorded == FLAGS.batch_per_async:
            training.batches_recorded = 0
            training.current_batch = FIRST_BATCH
        state.count = 0

if not FLAGS.async:
    batch_td_errror = tf.concat(all_td_error_sum, axis=0)
    accum_td_error = accum_value(batch_td_errror)
    init_vars()

    ops_batch_upload = [
        er.states[ph.batch_idx].assign(ph.state),
        er.actions[ph.batch_idx].assign(ph.action),
        er.rewards[ph.batch_idx].assign(ph.reward)
    ]

    if SAVE_SUMMARIES:
        merged = tf.summary.merge_all()
        apply_accum_ops += [merged]
        train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)

    for s in sorted(stats.__dict__.keys()):
        accum_ops.append(tf.concat([[r] for r in stats.__dict__[s]], 0))

def train_accum_batch():
    batch_mtime = training.batches_mtime.get(training.current_batch)
    paths = batch_paths(training.current_batch)
    batch_idx = 0 if ER_BATCHES==1 else FLAGS.batch_keep+training.current_batch
    while True:
        try:
            mtime = os.path.getmtime(paths[0])
            if ER_BATCHES == 1 or mtime != batch_mtime:
                b = mmap_batch(paths, 'r')
                feed_dict = {
                    ph.state: b.states,
                    ph.action: b.actions,
                    ph.reward: b.rewards,
                    ph.batch_idx: batch_idx}
                sess.run(ops_batch_upload, feed_dict=feed_dict)
                # imshow([er.states[batch_idx][10][:,:,i].eval() for i in range(STATE_FRAMES)])
                training.batches_mtime[training.current_batch] = mtime
            break
        except:
            print('Waiting for batch %i...' % training.current_batch)
            time.sleep(0.1)

    r = sess.run(accum_ops, feed_dict={ph.enable_tdc: epoch.prev_td_error > 0., ph.batch_idx: batch_idx})
    sys.stdout.write('Batch %i/(%i+%i)  ' % (training.current_batch, FLAGS.batch_keep, FLAGS.batch_async))
    if app.policy_index != -1:
        for i,s in enumerate(reversed(sorted(stats.__dict__.keys()))):
            sys.stdout.write(s+': ' + ('\n'if s[0]=='_'else'') + str(r[-i-1][app.policy_index])+'  ')
        print('')

def train_apply_gradients():
    epoch.prev_td_error = epoch.td_error
    epoch.td_error = sess.run(accum_td_error)[0]
    epoch.count += 1
    training.learning_rate *= 0.5 if epoch.td_error > epoch.prev_td_error else 2**0.5
    training.learning_rate = max(FLAGS.learning_rate, min(1e-1, training.learning_rate))
    r = sess.run(apply_accum_ops, feed_dict={ph.learning_rate: training.learning_rate})
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
        if FLAGS.async or not training.enable or training.batches_recorded < FLAGS.batch_keep:
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
