import tensorflow as tf, numpy as np
from tensorflow.contrib import rnn
import sys, os, multiprocessing, time, math
from utils import *
from pprint import pprint

# python rltf.py --batch_keep 6 --batch_async 20 --record 0
# for i in {1..4}; do python rltf.py --async $i --batch_per_async 5 & sleep 1; done

flags = tf.app.flags
flags.DEFINE_integer('async', 0, 'ID of async agent that accumulates gradients on server')
flags.DEFINE_integer('batch_async', 10, 'Batches to wait for from async agents')
flags.DEFINE_integer('batch_per_async', 1, 'Batches recorded per async agent')
flags.DEFINE_integer('batch_keep', 5, 'Batches recorded from user actions')
flags.DEFINE_boolean('replay', False, 'Replay actions recorded in memmap array')
flags.DEFINE_boolean('recreate_states', False, 'Recreate kept states from saved raw frames')
flags.DEFINE_boolean('record', False, 'Record over kept batches')
flags.DEFINE_boolean('er_all', True, 'Keep all batches in ER memory')
flags.DEFINE_boolean('summary', True, 'Save summaries for Tensorboard')
flags.DEFINE_boolean('sample_actions', False, 'Sample actions, or use policy')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
FLAGS = flags.FLAGS

PORT, PROTOCOL = 'localhost:2222', 'grpc'
if not FLAGS.async:
    server = tf.train.Server({'local': [PORT]}, protocol=PROTOCOL, start=True)
sess = tf.InteractiveSession(PROTOCOL+'://'+PORT)

STATE_FRAMES = 2    # Frames an action is repeated for, combined into a state
ADJACENT_STATES = 4 # Combined adjacent states, fed to value/policy function

EXTRA_ACTIONS = []
ENV_NAME = os.getenv('ENV')
if not ENV_NAME:
    raise Exception('Missing ENV environment variable')
if ENV_NAME == 'CarRacing-v0':
    import gym.envs.box2d
    car_racing = gym.envs.box2d.car_racing
    car_racing.WINDOW_W = 800 # Default is huge
    car_racing.WINDOW_H = 600
    EXTRA_ACTIONS = [
        [1,  0, 0],
        [-1, 0, 0],
        [0,  0, 0],
        [0,  1, 0],
        [1,  1, 0],
        [-1, 1, 0],
        [0,  0, 0.8],
    ] if 1 else []
elif ENV_NAME == 'FlappyBird-v0':
    import gym_ple # [512, 288]
elif 'Bullet' in ENV_NAME:
    import pybullet_envs

import gym
env = gym.make(ENV_NAME)
env._max_episode_steps = None # Disable step limit
envu = env.unwrapped

ACTION_DIM = (env.action_space.shape or [env.action_space.n])[0]
ACTION_DISCRETE = not env.action_space.shape
def onehot_vector(action_index): return [1. if action_index == i else 0. for i in range(ACTION_DIM)]
if ACTION_DISCRETE:
    ACTION_COMBIS = [onehot_vector(a) for a in range(ACTION_DIM)]
else:
    ACTION_COMBIS = EXTRA_ACTIONS
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
ER_BATCH_STEPS = ER_BATCH_SIZE-1
ER_BATCHES = FLAGS.batch_keep + FLAGS.batch_async if (FLAGS.er_all and not FLAGS.async) else 1

training = Struct(enable=True, batches_recorded=0, batches_mtime={}, temp_batch=None)
epoch = Struct(count=0)
app = Struct(policy_index=0, quit=False)

def batch_paths(batch_num, path=None):
    if not path:
        path = ENV_NAME + '_' + str(STATE_FRAMES)
    os.system('mkdir -p batches')
    path = 'batches/' + path + '_%i_%s.mmap'
    return {key: path % (batch_num, key) for key in ['rawframes', 'states', 'actions', 'rewards']}

REWARDS_GLOBAL = REWARDS_ALL = 1
def mmap_batch(paths, mode, only_actions=False, rawframes=True):
    batch = Struct(actions=np.memmap(paths['actions'], DTYPE.name, mode, shape=(ER_BATCH_SIZE, ACTION_DIM)))
    if only_actions:
        return batch.actions
    batch.states = np.memmap(paths['states'], DTYPE.name, mode, shape=(ER_BATCH_SIZE,) + tuple(STATE_DIM))
    batch.rewards = np.memmap(paths['rewards'], DTYPE.name, mode, shape=(ER_BATCH_SIZE, REWARDS_GLOBAL))
    batch.arrays = [batch.states, batch.actions, batch.rewards]
    if rawframes:
        batch.rawframes = np.memmap(paths['rawframes'], DTYPE.name, mode, shape=(ER_BATCH_SIZE, STATE_FRAMES) + tuple(FRAME_DIM))
        batch.arrays.append(batch.rawframes)
    return batch

ph = Struct(learning_rate=tf.placeholder(tf.float32, ()),
    enable_tdc=tf.placeholder(tf.bool, ()),
    batch_idx=tf.placeholder(tf.int32, ()),
    states=tf.placeholder(tf.float32, [None] + STATE_DIM),
    action=tf.placeholder(tf.float32, [None, ACTION_DIM]),
    reward=tf.placeholder(tf.float32, [None, REWARDS_GLOBAL]),
    frame=tf.placeholder(tf.float32, [None, STATE_FRAMES] + FRAME_DIM))
opt_td = tf.train.GradientDescentOptimizer(ph.learning_rate)
opt_policy = tf.train.GradientDescentOptimizer(ph.learning_rate/10)

if FLAGS.async:
    FIRST_BATCH = (FLAGS.async-1)*FLAGS.batch_per_async
else:
    def init_vars(): sess.run(tf.global_variables_initializer())

    if FLAGS.record or FLAGS.replay or FLAGS.recreate_states:
        # Record new arrays
        app.policy_index = -1
    else:
        training.batches_recorded = FLAGS.batch_keep
training.current_batch = FIRST_BATCH

with tf.name_scope('experience_replay'):
    er = Struct(
        states= tf.Variable(tf.zeros([ER_BATCHES, ER_BATCH_SIZE] + STATE_DIM), False, name='states'),
        actions=tf.Variable(tf.zeros([ER_BATCHES, ER_BATCH_STEPS, ACTION_DIM]), False, name='actions'),
        rewards=tf.Variable(tf.zeros([ER_BATCHES, ER_BATCH_STEPS, REWARDS_GLOBAL]), False, name='rewards'))
    er.batch_states = er.states[ph.batch_idx]
    er.batch_action = er.actions[ph.batch_idx]
    er.batch_reward = er.rewards[ph.batch_idx]
    all_rewards = [er.batch_reward[:, r] for r in range(REWARDS_ALL)]

    # Combine adjacent states for more temporal context
    states = er.batch_states
    er.adjacent_states = []
    for i in range(ADJACENT_STATES):
        er.adjacent_states.append(states)
        states = tf.concat([tf.expand_dims(tf.zeros(STATE_DIM), 0), states[:-1]], 0)
    er.adjacent_states = tf.concat(er.adjacent_states, -1)

if CONV_NET:
    frame_to_state = tf.reshape(ph.frame, [-1] + FRAME_DIM) # Move STATE_FRAMES into batches
    if RESIZE:
        frame_to_state = tf.image.resize_images(frame_to_state, RESIZE, tf.image.ResizeMethod.AREA)
    if FRAME_LCN:
        frame_to_state = local_contrast_norm(frame_to_state, GAUSS_W)
        frame_to_state = tf.reduce_max(frame_to_state, axis=-1)
    else:
        frame_to_state = tf.reduce_mean(frame_to_state/255., axis=-1) # Convert to grayscale
    frame_to_state = tf.reshape(frame_to_state, [-1, STATE_FRAMES] + STATE_DIM[:-1])
    frame_to_state = tf.transpose(frame_to_state, [0, 2, 3, 1]) # Move STATE_FRAMES into channels
else:
    frame_to_state = tf.reshape(ph.frame, [-1] + STATE_DIM)

ops = Struct(accum=[], apply_accum=[], zero_accum=[])
def accum_value(value):
    accum = tf.Variable(tf.zeros_like(value),trainable=False)
    ops.accum.append(accum.assign_add(value))
    ops.zero_accum.append(accum.assign(tf.zeros_like(value)))
    return accum
def accum_gradient(grads, opt):
    global_norm = None
    grads, vars = zip(*grads)
    grads = [accum_value(g) for g in grads]
    # Clip gradients by global norm to prevent destabilizing policy
    grads,global_norm = tf.clip_by_global_norm(grads, 50.)
    grads = zip(grads, vars)
    ops.apply_accum.append(opt.apply_gradients(grads))
    return global_norm

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

def make_dense_layers(x, scope='', outputs=200, layers=1, activation=tf.nn.relu):
    _scope = tf.contrib.framework.get_name_scope() + '/dense' + scope
    for l in range(layers or 1):
        dense = tf.layers.Dense(outputs, _scope=_scope+str(l))
        x = [dense.apply(i) for i in x]
        if activation:
            x = [activation(i) for i in x]
    return x
def make_dense_output(x, outputs, scope='output'):
    return make_dense_layers(x, scope, outputs, activation=None)

def make_conv_net(x):
    LAYERS = [
        # DQN & A3C
        (64, 8, 4),#(32, 8, 4),
        (64, 4, 2),
        (64, 3, 1)]
    _scope = tf.contrib.framework.get_name_scope() + '/conv'
    print(x[0].shape)
    for l in range(len(LAYERS) if CONV_NET else 0):
        filters, width, stride = LAYERS[l]
        conv = tf.layers.Conv2D(filters, width, stride, activation=tf.nn.relu,
            _scope=_scope+str(l))
        x = [conv.apply(i) for i in x]
        #x = [tf.nn.local_response_normalization(i) for i in x]
        print(x[0].shape)
    x = [tf.layers.flatten(i) for i in x]
    print(x[0].shape)
    x = [tf.contrib.layers.layer_norm(i, center=True, scale=True, scope=_scope, reuse=tf.AUTO_REUSE) for i in x]
    return x

def make_lstm_layer(x):
    cell = rnn.BasicLSTMCell(LSTM_NODES)
    scope = tf.contrib.framework.get_name_scope()
    stacked, _ = tf.nn.dynamic_rnn(cell, tf.stack(x[:1]), dtype=tf.float32, scope=scope)
    x = tf.unstack(stacked) + [x[1]]
    with tf.name_scope(str(FLAGS.async)):
        lstm_saved_state = tf.contrib.rnn.LSTMStateTuple(
            *[tf.Variable(s, trainable=False) for s in cell.zero_state(1, DTYPE)])
        sess.run(tf.variables_initializer(lstm_saved_state))
        stacked, final_state = tf.nn.dynamic_rnn(cell, tf.stack(x[-1:]), initial_state=lstm_saved_state, scope=scope)
        x = [x[0]] + tf.unstack(stacked)
        lstm_save_ops = [lstm_saved_state[i].assign(final_state[i]) for i in range(2)]
    return x, lstm_save_ops

def make_next_state_pair(x):
    x = [x[0][:-1], x[0][1:], x[1]]
    return x

GAMMA = 0.99
LSTM_NODES = 256
LSTM_LAYER = False
SHARED_LAYERS = True
MAX_NEXT_VALUE = True

allac = []
stats = Struct(reward_sum=[], td_error=[], _policy_minmax=[])
def make_acrl():
    input_states = [er.adjacent_states,
        tf.expand_dims(tf.concat([frame_to_state[i] for i in range(ADJACENT_STATES)], -1), 0)]

    ac = Struct(lstm_save_ops=[])
    allac.append(ac)
    with tf.name_scope('policy'):
        net = make_dense_layers(make_conv_net(input_states))
        if LSTM_LAYER: net, ac.lstm_save_ops = make_lstm_layer(net)
        net = make_next_state_pair(net)
        shared_net = net
        shared_weights = scope_vars()
        net = make_dense_output(net, ACTION_DIM)

        policy = [tf.nn.softmax(n) if POLICY_SOFTMAX else tf.tanh(n) for n in net]
        ac.policy = policy[0]
        ac.ph_policy = policy[2]
        stats._policy_minmax.append(tf.concat([
            [tf.reduce_min(policy[0], axis=0)],
            [tf.reduce_max(policy[0], axis=0)]], axis=0))

    def maximize_value_pre(idx): # Maximize next-state value
        if not MAX_NEXT_VALUE or not ACTION_COMBIS: return
        a = tf.expand_dims(tf.constant(ACTION_COMBIS), 0)
        a = tf.tile(a, [ER_BATCH_STEPS, 1, 1])
        num_combis = len(ACTION_COMBIS)
        if 1:
            num_combis += 1 # Add the current policy action as well
            policy_action = tf.expand_dims(actions[idx], 1)
            a = tf.concat([a, policy_action], 1)
        n = tf.expand_dims(net[idx], 1)
        n = tf.tile(n, [1, num_combis, 1])

        a = tf.reshape(a, [ER_BATCH_STEPS*num_combis, -1])
        n = tf.reshape(n, [ER_BATCH_STEPS*num_combis, -1])
        net[idx],actions[idx] = n,a

    def maximize_value_post(idx):
        if not MAX_NEXT_VALUE or not ACTION_COMBIS: return
        n = net[idx]
        n = tf.reshape(n, [ER_BATCH_STEPS, int(n.shape[0])/ER_BATCH_STEPS, -1])
        n = tf.reduce_max(n, 1)
        net[idx] = n

    with tf.name_scope('value'):
        if SHARED_LAYERS:
            net = shared_net
        else:
            net = make_dense_layers(make_conv_net(input_states))
            if LSTM_LAYER: net, _ = make_lstm_layer(net)
            net = make_next_state_pair(net)
        net = [net[0], net[0], net[1], net[2]]
        actions = [policy[0], er.batch_action, policy[1], policy[2]]

        maximize_value_pre(2)
        net = [tf.concat([n,a], 1) for n,a in zip(net,actions)]
        net = make_dense_layers(net, scope='layer2', activation=tf.nn.leaky_relu)
        net = make_dense_output(net, 1)
        maximize_value_post(2)

        [ac.state_value, ac.q_value, next_state_value, ac.ph_policy_value] = [n[:,0] for n in net]
        ac.q_grad = tf.gradients(ac.state_value, ac.policy)[0]
        ac.ph_policy_value_grad = tf.gradients(ac.ph_policy_value, ac.ph_policy)[0]

    def multi_step(recurse, i=0):# recurse==0 is 1-step TD
        z = tf.zeros(i)
        next = tf.concat([next_state_value[i:], z], 0)
        return tf.concat([all_rewards[r][i:], z], 0) + GAMMA*(tf.where(
            tf.range(i, ER_BATCH_STEPS+i) < ER_BATCH_STEPS-1,
            tf.maximum(next, multi_step(recurse, i+1)),
            #multi_step(recurse, i+1),
            next) if i < recurse else next)
    ac.step_1 = multi_step(0)
    ac.step_n = multi_step(40)
    ac.td_error = (0.9*ac.step_1+0.1*ac.step_n) - ac.q_value

    POLICY_CLAMP = 1.-2e-2
    q_grad = tf.where(ac.policy>POLICY_CLAMP, tf.minimum(0., ac.q_grad),
             tf.where(ac.policy<-POLICY_CLAMP, tf.maximum(0., ac.q_grad), ac.q_grad))
    repl = gradient_override(ac.policy, q_grad)
    grad = opt_policy.compute_gradients(repl, scope_vars('policy'))
    ac.global_norm = accum_gradient(grad, opt_policy)

    # TD error with gradient correction (TDC)
    weights = scope_vars('value') + (shared_weights if SHARED_LAYERS else [])
    repl = gradient_override(ac.q_value, ac.td_error)
    grad_s = opt_td.compute_gradients(repl, weights)
    repl = gradient_override(next_state_value, -GAMMA*ac.td_error)
    grad_s2 = opt_td.compute_gradients(repl, weights)
    for i in range(len(grad_s)):
        (g, w), g2 = grad_s[i], grad_s2[i][0]
        if g2 is not None: g += tf.where(ph.enable_tdc, g2, tf.zeros(g2.shape))
        grad_s[i] = (g, w)
    accum_gradient(grad_s, opt_td)

    stats.reward_sum.append(tf.reduce_sum(all_rewards[r]))
    stats.td_error.append(tf.reduce_sum(ac.td_error**2))

for r in range(REWARDS_ALL):
    with tf.name_scope('ac_' + str(r)):
        make_acrl()

state = Struct(frames=np.zeros([ADJACENT_STATES, STATE_FRAMES] + FRAME_DIM),
               count=0, last_obs=None, last_pos_reward=0,
               done=True, next_reset=False)

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
            FLAGS.sample_actions ^= True
        elif k==ord('t'):
            training.enable ^= True
        elif k==ord('r'):
            state.next_reset = True
        else:
            return
        print(dict(
            policy_index=app.policy_index,
            training_enable=training.enable,
            sample_actions=FLAGS.sample_actions))

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

action = Struct(to_take=None, policy=None, keyboard=setup_key_actions())
def step_to_frames():
    def softmax(x):
        e_x = np.exp(x - x.max())
        return e_x / e_x.sum()

    a = action.keyboard[:ACTION_DIM]
    if ACTION_DISCRETE:
        a = int(a[0]+1.)

    if state.count > 0:
        if app.policy_index != -1:
            a = action.policy[:]
            if ACTION_DISCRETE:
                if FLAGS.sample_actions:
                    # Sample from softmax probabilities
                    prob = softmax(a)
                    a = np.random.choice(range(ACTION_DIM), p=prob)
                else:
                    a = np.argmax(a)
            else:
                if FLAGS.sample_actions:
                    dim = np.random.choice(range(ACTION_DIM), p=softmax(action.policy_value_grad))
                    a[dim] += 0.1*np.random.rand() * (1. if action.policy_value_grad[dim] > 0. else -1.)

    env_action = a
    if ACTION_DISCRETE:
        action.to_save = onehot_vector(env_action)
    else:
        action.to_save = env_action
    #if FLAGS.async == 1: print(action.to_save)
    obs = state.last_obs
    reward_sum = 0.
    state.frames[1:] = state.frames[:-1]
    for frame in range(STATE_FRAMES):
        state.done |= state.next_reset
        if state.done:
            state.next_reset = False
            # New episode
            env.seed(0) # Same track everytime
            obs = env.reset()
        env.render()
        #imshow([obs, test_lcn(obs, sess)[0]])
        state.frames[0, frame] = obs

        obs, reward, state.done, info = env.step(env_action)
        if ENV_NAME == 'MountainCar-v0':
            # Mountain car env doesnt give any +reward
            reward = 1. if state.done else 0.
        elif ENV_NAME == 'CarRacing-v0':
            if state.last_pos_reward > 20:
                state.done = True # Reset track if spinning
                reward = -1000
        state.last_pos_reward = 0 if reward>0. or state.done else state.last_pos_reward+1
        reward_sum += reward
    state.last_obs = obs
    return [reward_sum]

ops_single_step = [frame_to_state[0]] + [
    i for sublist in [
        [ac.ph_policy[0],# Policy from uploaded state,
        ac.ph_policy_value,
        ac.ph_policy_value_grad[0],
        ] + ac.lstm_save_ops for ac in allac]
    for i in sublist]
def append_to_batch():
    save_paths = batch_paths(training.current_batch)
    if not FLAGS.async and FLAGS.recreate_states:
        if not state.count:
            training.saved_batch = mmap_batch(save_paths, 'r')
        batch = training.saved_batch
        state.frames[0] = batch.rawframes[state.count]
        save_reward = batch.rewards[state.count]
        save_action = batch.actions[state.count]
    else:
        save_reward = step_to_frames()
        save_action = action.to_save

    ret = sess.run(ops_single_step, feed_dict={ph.frame: state.frames})
    save_state = ret[0]
    action.policy = ret[1]
    action.policy_value = ret[2]
    action.policy_value_grad = ret[3]

    temp_paths = batch_paths(FLAGS.async, 'temp')
    if not training.temp_batch:
        training.temp_batch = mmap_batch(temp_paths, 'w+')
    batch = training.temp_batch
    batch.rawframes[state.count] = state.frames[0]
    batch.states[state.count] = save_state
    batch.rewards[state.count] = save_reward
    batch.actions[state.count] = save_action
    # imshow([save_state[:,:,i] for i in range(STATE_FRAMES)])

    state.count += 1
    if state.count == ER_BATCH_SIZE:
        if FLAGS.async or training.batches_recorded < FLAGS.batch_keep:
            print('Replacing batch #%i' % training.current_batch)
            for a in batch.arrays: del a
            training.temp_batch = None

            # Rename async batch files into server's ER batches.
            for k in save_paths.keys():
                src = temp_paths[k]
                dst = save_paths[k]
                os.system('rm -f ' + dst)
                os.system('mv ' + src + ' ' + dst)

        training.batches_recorded += 1
        training.current_batch += 1
        if FLAGS.async and training.batches_recorded == FLAGS.batch_per_async:
            training.batches_recorded = 0
            training.current_batch = FIRST_BATCH
        state.count = 0

if not FLAGS.async:
    ops_batch_upload = [
        er.batch_states.assign(ph.states),
        er.batch_action.assign(ph.action),
        er.batch_reward.assign(ph.reward)
    ]

    ops.train = [
        accum_value(tf.concat([stats.td_error], 0)),
        accum_value(tf.concat([stats.reward_sum], 0)),
        tf.concat([[ac.global_norm] for ac in allac], 0)
    ] + ops.apply_accum
    init_vars()

    for s in sorted(stats.__dict__.keys()):
        ops.accum.append(tf.concat([[r] for r in stats.__dict__[s]], 0))

    if FLAGS.summary:
        merged = tf.summary.merge_all()
        #ops.train += [merged]
        train_writer = tf.summary.FileWriter('/tmp/tf', sess.graph)

def train_accum_batch():
    batch_mtime = training.batches_mtime.get(training.current_batch)
    paths = batch_paths(training.current_batch)
    batch_idx = 0 if ER_BATCHES==1 else FLAGS.batch_keep+training.current_batch
    while True:
        try:
            mtime = os.path.getmtime(paths['states'])
            if ER_BATCHES == 1 or mtime != batch_mtime:
                b = mmap_batch(paths, 'r', rawframes=False)
                feed_dict = {
                    ph.states: b.states,
                    ph.action: b.actions[1:],
                    ph.reward: b.rewards[1:],
                    ph.batch_idx: batch_idx}
                sess.run(ops_batch_upload, feed_dict=feed_dict)
                training.batches_mtime[training.current_batch] = mtime
            break
        except:
            print('Waiting for batch %i...' % training.current_batch)
            time.sleep(0.1)

    feed_dict = {ph.enable_tdc: True, ph.batch_idx: batch_idx}
    # imshow([er.adjacent_states[20][:,:,i].eval(feed_dict) for i in range(ADJACENT_STATES*STATE_FRAMES)])
    r = sess.run(ops.accum, feed_dict)
    if training.current_batch + FLAGS.batch_keep > 12:
        return # No more screen space
    sys.stdout.write('Batch %i/(%i+%i)  ' % (training.current_batch, FLAGS.batch_keep, FLAGS.batch_async))
    if app.policy_index != -1:
        for i,s in enumerate(reversed(sorted(stats.__dict__.keys()))):
            sys.stdout.write(s+': ' + ('\n'if s[0]=='_'else'') + str(r[-i-1][app.policy_index])+'  ')
        print('')

def train_apply_gradients():
    r = sess.run(ops.train, feed_dict={ph.learning_rate: FLAGS.learning_rate})
    epoch.count += 1
    sess.run(ops.zero_accum)
    os.system('clear')
    pprint(dict(
        policy_index=app.policy_index,
        epoch_count=epoch.count,
        epoch_td_error=          r[0][app.policy_index],
        epoch_reward_sum=        r[1][app.policy_index],
        epoch_policy_global_norm=r[2][app.policy_index],
        learning_rate='%.6f' % FLAGS.learning_rate))

    if 0:#FLAGS.summary:
        summary = r[-1]
        train_writer.add_summary(summary, epoch.count)

def rl_loop():
    while not app.quit:
        if FLAGS.async or not training.enable or training.batches_recorded < FLAGS.batch_keep:
            append_to_batch()
            continue

        if training.current_batch < LAST_BATCH:
            train_accum_batch()
            training.current_batch += 1
        else:
            env.render() # Render needed for keyboard events
            train_apply_gradients()
            training.current_batch = FIRST_BATCH
rl_loop()
