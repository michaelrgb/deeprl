import tensorflow as tf, numpy as np
from tensorflow.contrib import rnn
import sys, os, multiprocessing, time
from utils import *
from pprint import pprint

flags = tf.app.flags
# for i in {1..10}; do python rltf.py --async $i --batch_per_async 1 & sleep 1; done
flags.DEFINE_integer('async', 0, 'ID of async agent that accumulates gradients on server')
flags.DEFINE_integer('batch_async', 10, 'Batches to wait for from async agents')
flags.DEFINE_integer('batch_per_async', 1, 'Batches recorded per async agent')
flags.DEFINE_integer('batch_keep', 5, 'Batches recorded from user actions')
flags.DEFINE_boolean('replay', False, 'Replay actions recorded in memmap array')
flags.DEFINE_boolean('record', False, 'Record over kept batches')
flags.DEFINE_boolean('er_all', False, 'Keep all batches in ER memory')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
FLAGS = flags.FLAGS

PORT, PROTOCOL = 'localhost:2222', 'grpc'
if not FLAGS.async:
    server = tf.train.Server({'local': [PORT]}, protocol=PROTOCOL, start=True)
sess = tf.InteractiveSession(PROTOCOL+'://'+PORT)

GAMMA = 0.99
STATE_FRAMES = 2
HIDDEN_LAYERS = 3
HIDDEN_NODES = 128
LSTM_NODES = 128
LSTM_LAYER = True
SHARED_LAYERS = False

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
env._max_episode_steps = None # Disable step limit
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

training = Struct(enable=True, batches_recorded=0, batches_mtime={}, temp_batch=None)
epoch = Struct(count=0)
app = Struct(policy_index=0, sample_actions=False, quit=False)

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
    states=tf.placeholder(tf.float32, [None] + STATE_DIM),
    action=tf.placeholder(tf.float32, [None, ACTION_DIM]),
    reward=tf.placeholder(tf.float32, [None, REWARDS_GLOBAL]),
    frame=tf.placeholder(tf.float32, [None, STATE_FRAMES] + FRAME_DIM))
opt_td = tf.train.AdamOptimizer(ph.learning_rate, epsilon=0.1)
opt_policy = tf.train.AdamOptimizer(ph.learning_rate/10, epsilon=0.1)

if FLAGS.async:
    FIRST_BATCH = (FLAGS.async-1)*FLAGS.batch_per_async
else:
    def init_vars(): sess.run(tf.global_variables_initializer())

    if FLAGS.record or FLAGS.replay:
        # Record new arrays
        app.policy_index = -1
    else:
        training.batches_recorded = FLAGS.batch_keep
training.current_batch = FIRST_BATCH

with tf.name_scope('experience_replay'):
    er = Struct(
        states= tf.Variable(tf.zeros([ER_BATCHES, ER_BATCH_SIZE] + STATE_DIM), False, name='states'),
        actions=tf.Variable(tf.zeros([ER_BATCHES, ER_BATCH_SIZE-1, ACTION_DIM]), False, name='actions'),
        rewards=tf.Variable(tf.zeros([ER_BATCHES, ER_BATCH_SIZE-1, REWARDS_GLOBAL]), False, name='rewards'))
    er.batch_states = er.states[ph.batch_idx]
    er.batch_action = er.actions[ph.batch_idx]
    er.batch_reward = er.rewards[ph.batch_idx]
    all_rewards = [er.batch_reward[:, r] for r in range(REWARDS_ALL)] + []

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
    global_norm = None
    grads, vars = zip(*grads)
    grads = [accum_value(g) for g in grads]
    if True:
        # Clip gradients by global norm to prevent destabilizing policy
        grads,global_norm = tf.clip_by_global_norm(grads, 50.)
    grads = zip(grads, vars)
    apply_accum_ops.append(opt.apply_gradients(grads))
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

all_policy = []
stats = Struct(reward_sum=[], td_error=[], on_policy_count=[], max_advantage=[], _policy_minmax=[])

def make_dense_layers(x, outputs):
    for l in range(HIDDEN_LAYERS):
        dense = tf.layers.Dense(HIDDEN_NODES, tf.nn.leaky_relu,
            _scope=tf.contrib.framework.get_name_scope() + '/fully%i' % l)
        x = [dense.apply(i) for i in x]
    if outputs:
        dense = tf.layers.Dense(outputs, None,
            _scope=tf.contrib.framework.get_name_scope() + '/fully%i' % HIDDEN_LAYERS)
    return [dense.apply(i) for i in x]

def make_conv_net(x):
    filters = 32
    CONV_LAYERS = 3
    for l in range(CONV_LAYERS if CONV_NET else 0):
        conv = tf.layers.Conv2D(filters, 8, 2,
            _scope=tf.contrib.framework.get_name_scope() + '/conv%i' % l)
        x = [conv.apply(i) for i in x]
        print(x[0].shape)
    x = [tf.layers.flatten(i) for i in x]
    print(x[0].shape)

    # Add layers before LSTM
    if not CONV_NET:
        with tf.name_scope('pre_lstm'): x = make_dense_layers(x, None)
    return x

def make_next_state_pair(x):
    x = [x[0][:-1], x[0][1:], x[1]]
    return x

def make_acrl():
    input_states = [er.batch_states, frame_to_state]
    p = Struct()
    with tf.name_scope('shared'):
        shared_layer = make_conv_net(input_states)
        if LSTM_LAYER:
            cell = rnn.BasicLSTMCell(LSTM_NODES)
            scope = tf.contrib.framework.get_name_scope()
            stacked, _ = tf.nn.dynamic_rnn(cell, tf.stack(shared_layer[:1]), dtype=tf.float32, scope=scope)
            shared_layer[:1] = tf.unstack(stacked)
            with tf.name_scope(str(FLAGS.async)):
                p.lstm_saved_state = tf.contrib.rnn.LSTMStateTuple(
                    *[tf.Variable(s, trainable=False) for s in cell.zero_state(1, DTYPE)])
                sess.run(tf.variables_initializer(p.lstm_saved_state))
                stacked, final_state = tf.nn.dynamic_rnn(cell, tf.stack(shared_layer[-1:]), initial_state=p.lstm_saved_state, scope=scope)
                shared_layer[-1:] = tf.unstack(stacked)
                p.lstm_save_ops = [p.lstm_saved_state[i].assign(final_state[i]) for i in range(2)]
        shared_layer = make_next_state_pair(shared_layer)
        shared_weights = scope_vars()

    with tf.name_scope('policy'):
        policy = make_dense_layers(shared_layer, ACTION_DIM)
        policy = [tf.nn.softmax(i) if POLICY_SOFTMAX else tf.tanh(i) for i in policy]
        p.policy = policy[-1]
        all_policy.append(p)
        stats._policy_minmax.append(tf.concat([
            [tf.reduce_min(policy[0], axis=0)],
            [tf.reduce_max(policy[0], axis=0)]], axis=0))

    if FLAGS.async:
        return

    if SHARED_LAYERS:
        value_layer = shared_layer
    else:
        with tf.name_scope('non_shared'):
            value_layer = make_conv_net(input_states)
            if LSTM_LAYER:
                cell = rnn.BasicLSTMCell(LSTM_NODES)
                scope = tf.contrib.framework.get_name_scope()
                stacked, _ = tf.nn.dynamic_rnn(cell, tf.stack(value_layer[:1]), dtype=tf.float32, scope=scope)
                value_layer[:1] = tf.unstack(stacked)
                with tf.name_scope(str(FLAGS.async)):
                    # Not used currently
                    stacked, _ = tf.nn.dynamic_rnn(cell, tf.stack(value_layer[-1:]), dtype=tf.float32, scope=scope)
                    value_layer[-1:] = tf.unstack(stacked)
            value_layer = make_next_state_pair(value_layer)
            shared_weights = scope_vars()

    with tf.name_scope('value'):
        value_layer =   [value_layer[0], value_layer[0],  value_layer[1]]
        value_actions = [policy[0],      er.batch_action, policy[1]]
        # Include actions as input to dense layers
        value_layer = [tf.concat([value_layer[i], value_actions[i]], 1) for i in range(3)]

        value_layer = make_dense_layers(value_layer, 1+ACTION_DIM)
        avg_value = [i[:,0] for i in value_layer]
        adv_direction = [i[:,1:] for i in value_layer]

        [state_value, q_value, next_state_value] = [(tf.reduce_sum(
            adv_direction[i]*value_actions[i], 1) if 1 else 0) + avg_value[i] for i in range(3)]

    policy_action = tf.one_hot(tf.argmax(policy[0], 1), ACTION_DIM) if ACTION_DISCRETE else policy[0]
    is_on_policy = tf.reduce_sum(tf.abs(er.batch_action-policy_action), 1) < 0.1
    def multi_step(max_recurse, i=0):
        z = tf.zeros(i)
        next = tf.concat([next_state_value[i:], z], 0)
        return tf.concat([all_rewards[r][i:], z], 0) + GAMMA*(tf.where(
            tf.concat([is_on_policy[i+1:], tf.zeros(i+1, dtype=tf.bool)], 0),
            multi_step(max_recurse, i+1),
            next) if i < max_recurse else next)
    target_value = multi_step(0)# 0 is 1-step TD
    td_error = target_value - q_value

    adv = target_value - state_value
    adv = tf.maximum(0., adv)
    if POLICY_SOFTMAX:
        prob = tf.reduce_sum(policy[0]*er.batch_action, 1)
        log_prob = tf.log(tf.clip_by_value(prob, 1e-20, 1-1e-20))
        adv = tf.where(prob < 1e-2, tf.maximum(0., adv),
              tf.where(prob > 1-1e-2, tf.minimum(0., adv), adv))
    else:
        off_policy = policy[0]-er.batch_action
        log_prob = -(off_policy)**2
        adv = tf.tile(tf.expand_dims(adv, 1), [1, ACTION_DIM])

    repl = gradient_override(log_prob, adv)
    grad = opt_policy.compute_gradients(repl, scope_vars('policy')+scope_vars('shared'))
    p.global_norm = accum_gradient(grad, opt_policy)

    stats.max_advantage.append(tf.reduce_max(adv))
    stats.on_policy_count.append(tf.reduce_sum(tf.cast(is_on_policy, tf.int32)))
    stats.reward_sum.append(tf.reduce_sum(all_rewards[r]))
    stats.td_error.append(tf.reduce_sum(td_error**2))

    # TD error with gradient correction (TDC)
    weights = scope_vars('value')+shared_weights
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

state = Struct(frames=np.zeros([STATE_FRAMES] + FRAME_DIM),
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
            app.sample_actions ^= True
        elif k==ord('t'):
            training.enable ^= True
        elif k==ord('r'):
            state.next_reset = True
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

action = Struct(to_take=None, policy=None, keyboard=setup_key_actions())
def step_to_frames():
    obs = state.last_obs
    env_action = np.argmax(action.to_take) if ACTION_DISCRETE else action.to_take
    reward_sum = 0.
    for frame in range(STATE_FRAMES):
        state.done |= state.next_reset
        if state.done:
            state.next_reset = False
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
        elif ENV_NAME == 'CarRacing-v0':
            if state.last_pos_reward > 100:
                state.done = True # Reset track if spinning
                reward = -100
        state.last_pos_reward = 0 if reward>0. or state.done else state.last_pos_reward+1
        reward_sum += reward
    state.last_obs = obs
    return [reward_sum]

ops_single_step = [
    frame_to_state[0],
    all_policy[0].policy[0],# Policy from uploaded state,
] + ([op for p in all_policy for op in p.lstm_save_ops] if LSTM_LAYER else [])
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
    accum_td_error = accum_value(tf.concat([stats.td_error], 0))
    init_vars()

    ops_batch_upload = [
        er.batch_states.assign(ph.states),
        er.batch_action.assign(ph.action),
        er.batch_reward.assign(ph.reward)
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
                    ph.states: b.states,
                    ph.action: b.actions[1:],
                    ph.reward: b.rewards[1:],
                    ph.batch_idx: batch_idx}
                sess.run(ops_batch_upload, feed_dict=feed_dict)
                # imshow([er.batch_states[10][:,:,i].eval(feed_dict) for i in range(STATE_FRAMES)])
                training.batches_mtime[training.current_batch] = mtime
            break
        except:
            print('Waiting for batch %i...' % training.current_batch)
            time.sleep(0.1)

    r = sess.run(accum_ops, feed_dict={ph.enable_tdc: True, ph.batch_idx: batch_idx})
    sys.stdout.write('Batch %i/(%i+%i)  ' % (training.current_batch, FLAGS.batch_keep, FLAGS.batch_async))
    if app.policy_index != -1:
        for i,s in enumerate(reversed(sorted(stats.__dict__.keys()))):
            sys.stdout.write(s+': ' + ('\n'if s[0]=='_'else'') + str(r[-i-1][app.policy_index])+'  ')
        print('')

def train_apply_gradients():
    r = sess.run([accum_td_error,
                  tf.concat([[p.global_norm] for p in all_policy], 0)] + apply_accum_ops,
        feed_dict={ph.learning_rate: FLAGS.learning_rate})
    epoch.count += 1
    sess.run(zero_accum_ops)
    os.system('clear')
    pprint(dict(
        policy_index=app.policy_index,
        epoch_count=epoch.count,
        epoch_td_error=          r[0][app.policy_index],
        epoch_policy_global_norm=r[1][app.policy_index],
        learning_rate='%.6f' % FLAGS.learning_rate))

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
