import tensorflow as tf, numpy as np
from tensorflow.contrib import rnn
import sys, os, multiprocessing, time, math
from utils import *
from pprint import pprint

# python rltf.py --batch_keep 6 --batch_async 20 --record 0
# for i in {1..4}; do python rltf.py --async $i --batch_per_async 5 & sleep 1; done

flags = tf.app.flags
flags.DEFINE_integer('async', 0, 'ID of async agent that accumulates gradients on server')
flags.DEFINE_integer('batch_keep', 5, 'Batches recorded from user actions')
flags.DEFINE_integer('batch_async', 10, 'Batches to wait for from async agents')
flags.DEFINE_integer('batch_per_async', 10, 'Batches recorded per async agent')
flags.DEFINE_boolean('replay', False, 'Replay actions recorded in memmap array')
flags.DEFINE_boolean('recreate_states', False, 'Recreate kept states from saved raw frames')
flags.DEFINE_boolean('record', False, 'Record over kept batches')
flags.DEFINE_boolean('summary', True, 'Save summaries for Tensorboard')
flags.DEFINE_boolean('sample_action', False, 'Sample actions, or use policy')
flags.DEFINE_boolean('quantize', False, 'Quantize actions to list')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_float('tau', 1e-3, 'Target network update rate')
flags.DEFINE_integer('minibatch', 20, 'Minibatch size')
FLAGS = flags.FLAGS

PORT, PROTOCOL = 'localhost:2222', 'grpc'
if not FLAGS.async:
    server = tf.train.Server({'local': [PORT]}, protocol=PROTOCOL, start=True)
sess = tf.InteractiveSession(PROTOCOL+'://'+PORT)

STATE_FRAMES = 3    # Frames an action is repeated for, combined into a state

QUANT_ACTIONS = []
ENV_NAME = os.getenv('ENV')
if not ENV_NAME:
    raise Exception('Missing ENV environment variable')
if ENV_NAME == 'CarRacing-v0':
    import gym.envs.box2d
    car_racing = gym.envs.box2d.car_racing
    car_racing.WINDOW_W = 800 # Default is huge
    car_racing.WINDOW_H = 600
    QUANT_ACTIONS = [
        [1, 0, 0],
        [0.1, 0, 0],
        [-1,0, 0],
        [-0.1, 0, 0],
        [0, 1, 0],
        [0, 0.1, 0],
        [0, 0, 0],
        [0, 0, 0.8],
        [0, 0, 0.2],
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
if ACTION_DISCRETE: EVAL_ACTIONS = [onehot_vector(a) for a in range(ACTION_DIM)]
else: EVAL_ACTIONS = QUANT_ACTIONS
POLICY_SOFTMAX = ACTION_DISCRETE
FRAME_DIM = list(env.observation_space.shape)

CONV_NET = len(FRAME_DIM) == 3
STATE_DIM = FRAME_DIM[:]
if CONV_NET:
    if STATE_DIM[-1] == 3:
        STATE_DIM[-1] = 1
    FRAME_LCN = True
    RESIZE = [84, 84] # DQN
    if RESIZE:
        STATE_DIM[:2] = RESIZE
STATE_DIM[-1] *= STATE_FRAMES

FIRST_BATCH = -FLAGS.batch_keep
LAST_BATCH = FLAGS.batch_async
ER_BATCH_SIZE = 100
ER_BATCH_STEPS = ER_BATCH_SIZE-1

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
    states=tf.placeholder(tf.float32, [FLAGS.minibatch] + STATE_DIM),
    next_states=tf.placeholder(tf.float32, [FLAGS.minibatch] + STATE_DIM),
    actions=tf.placeholder(tf.float32, [FLAGS.minibatch, ACTION_DIM]),
    rewards=tf.placeholder(tf.float32, [FLAGS.minibatch, REWARDS_GLOBAL]),
    frame=tf.placeholder(tf.float32, [None, STATE_FRAMES] + FRAME_DIM))

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

training = Struct(enable=True, batches_recorded=0, batches_mtime={}, temp_batch=None)
app = Struct(policy_index=0, quit=False, epoch_count=0, print_action=False, show_state_image=False)
if FLAGS.async:
    FIRST_BATCH = (FLAGS.async-1)*FLAGS.batch_per_async
else:
    def init_vars(): sess.run(tf.global_variables_initializer())
    if FLAGS.record or FLAGS.replay or FLAGS.recreate_states:
        # Record new arrays
        app.policy_index = -1
    else:
        training.batches_recorded = FLAGS.batch_keep
training.append_batch = FIRST_BATCH

ops = Struct(per_batch=[], post_batch=[], per_epoch=[], post_epoch=[], post_step=[], new_trajectories=[])
def accum_value(value):
    accum = tf.Variable(tf.zeros_like(value),trainable=False)
    ops.per_batch.append(accum.assign_add(value))
    ops.post_epoch.append(accum.assign(tf.zeros_like(value)))
    return accum
def accum_gradient(grads, opt):
    global_norm = None
    grads, vars = zip(*grads)
    grads = [accum_value(g) for g in grads]
    # Clip gradients by global norm to prevent destabilizing policy
    grads,global_norm = tf.clip_by_global_norm(grads, 10.)
    grads = zip(grads, vars)
    ops.per_epoch.append(opt.apply_gradients(grads))
    return global_norm

# Custom gradients to pre-multiply weight gradients before they are aggregated across the batch.
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

def layer_batch_norm(x):
    _scope = tf.contrib.framework.get_name_scope()
    norm = tf.layers.BatchNormalization(_scope=_scope)
    return [norm.apply(x[i], training=i==layer_batch_norm.training_idx) for i in range(len(x))]

def make_conv_net(x):
    LAYERS = [
        # DQN & A3C
        (32, 8, 4),
        (64, 4, 2),
        (64, 3, 1)]
    print(x[0].shape)
    for l in range(len(LAYERS) if CONV_NET else 0):
        filters, width, stride = LAYERS[l]
        with tf.name_scope('conv'):
            _scope = tf.contrib.framework.get_name_scope()
            x = layer_batch_norm(x)
            conv = tf.layers.Conv2D(filters, width, stride, activation=tf.nn.relu, _scope=_scope)
            x = [conv.apply(n) for n in x]
        print(x[0].shape)
    x = [tf.layers.flatten(n) for n in x]
    print(x[0].shape)
    return x

def layer_dense(x, outputs, norm=True, activation=tf.nn.relu):
    with tf.name_scope('dense'):
        _scope = tf.contrib.framework.get_name_scope()
        if norm:
            x = layer_batch_norm(x)
        if outputs:
            dense = tf.layers.Dense(outputs, _scope=_scope)
            x = [dense.apply(n) for n in x]
        if activation:
            x = [activation(n) for n in x]
        return x

def layer_lstm(x):
    cell = rnn.BasicLSTMCell(LSTM_NODES)
    def save_state_ops(saved_state, final_state):
        return [saved_state[i].assign(final_state[i]) for i in range(2)]

    def lstm_state(key, batch_size):
        ret = tf.contrib.rnn.LSTMStateTuple(
            *[tf.Variable(s, trainable=False, collections=[None]) for s in cell.zero_state(batch_size, DTYPE)])
        sess.run(tf.variables_initializer(ret))
        return ret

    with tf.name_scope('lstm'):
        scope = tf.contrib.framework.get_name_scope()
        x = layer_batch_norm(x)
        with tf.name_scope(str(FLAGS.async)):
            if len(x) > 1:
                async_lstm_state = lstm_state('async', 1)
                output, state_after = tf.nn.dynamic_rnn(cell, tf.expand_dims(x[-1], 1), initial_state=async_lstm_state, scope=scope)
                x[-1] = output[:,0,:]

                ops.post_step += save_state_ops(async_lstm_state, state_after)

        minibatch_lstm_state = lstm_state('minibatch', FLAGS.minibatch)
        ops.new_trajectories += [s.assign(tf.zeros_like(s)) for s in minibatch_lstm_state]

        # [batch_size, max_time (i.e. 1), ...],
        output, lstm_next_state = tf.nn.dynamic_rnn(cell, tf.expand_dims(x[0], 1), initial_state=minibatch_lstm_state, scope=scope)
        x[0] = output[:,0,:]
        ops.post_batch += save_state_ops(minibatch_lstm_state, lstm_next_state)
    return x

GAMMA = 0.99
HIDDEN_1, HIDDEN_2 = 400, 300
LSTM_NODES = 600
opt_td = tf.train.AdamOptimizer(ph.learning_rate)
opt_policy = tf.train.AdamOptimizer(ph.learning_rate/10)

allac = []
stats = Struct(reward_sum=[], td_error=[], _policy_minmax=[])
def make_acrl():
    ac = Struct()
    allac.append(ac)
    def make_shared(net, lstm_layer=False):
        net = layer_dense(make_conv_net(net), HIDDEN_1)
        net = layer_lstm(net) if lstm_layer else net
        return net

    layer_batch_norm.training_idx = 0
    def make_policy(net):
        net = make_shared(net, True)
        net = layer_dense(net, HIDDEN_2)
        return layer_dense(net, ACTION_DIM,
            activation=tf.nn.softmax if POLICY_SOFTMAX else tf.tanh)
    with tf.name_scope('policy'):
        [ac.policy, ac.ph_policy] = make_policy([ph.states, frame_to_state])
        stats._policy_minmax.append(tf.concat([
            [tf.reduce_min(ac.policy, axis=0)],
            [tf.reduce_max(ac.policy, axis=0)]], axis=0))

    def multi_actions_pre(idx, batch_size=FLAGS.minibatch, include_policy=True):
        if not EVAL_ACTIONS: return
        a = tf.expand_dims(tf.constant(EVAL_ACTIONS, tf.float32), 0)
        a = tf.tile(a, [batch_size, 1, 1])
        num_actions = len(EVAL_ACTIONS)
        if include_policy:
            num_actions += 1
            policy_action = tf.expand_dims(actions[idx], 1)
            a = tf.concat([policy_action, a], 1)
        action_combis = a
        n = tf.expand_dims(net[idx], 1)
        n = tf.tile(n, [1, num_actions, 1])
        n = tf.reshape(n, [batch_size*num_actions, -1])
        a = tf.reshape(a, [batch_size*num_actions, -1])
        net[idx],actions[idx] = n,a
        return action_combis
    def multi_actions_inner(net, actions):
        n = layer_dense(net, HIDDEN_2)
        a = layer_dense(actions, HIDDEN_2, norm=False, activation=None)
        n = [n*a for n,a in zip(n, a)]
        return [n[:,0] for n in layer_dense(n, 1, norm=False, activation=None)]
    def multi_actions_post(n, batch_size=FLAGS.minibatch, reduce_max=False):
        if not EVAL_ACTIONS: return n
        n = tf.reshape(n, [batch_size, int(n.shape[0])/batch_size])
        n = tf.reduce_max(n, 1) if reduce_max else n
        return n

    with tf.name_scope('value'):
        net = make_shared([ph.states, frame_to_state])
        net = [net[0], net[0], net[1]]
        actions = [ac.policy, ph.actions, ac.ph_policy]
        multi_actions_pre(2, 1)
        [ac.state_value, ac.q_value, ph_policy_value] = multi_actions_inner(net, actions)
        # Q for all actions in agent's current state
        ac.ph_policy_value = multi_actions_post(ph_policy_value, 1)

    if FLAGS.async:
        return

    layer_batch_norm.training_idx = -1 # No UPDATE_OPS for target network
    with tf.name_scope('_policy'):
        next_state_policy = make_policy([ph.next_states])
    with tf.name_scope('_value'):
        net = make_shared([ph.next_states])
        [next_state_value] = multi_actions_inner(net, next_state_policy)

    ac.q_grad = tf.gradients(ac.state_value, ac.policy)[0]
    POLICY_CLAMP = 1.-2e-2
    q_grad = tf.where(ac.policy>POLICY_CLAMP, tf.minimum(0., ac.q_grad),
             tf.where(ac.policy<-POLICY_CLAMP, tf.maximum(0., ac.q_grad), ac.q_grad))
    repl = gradient_override(ac.policy, q_grad)
    grad = opt_policy.compute_gradients(repl, scope_vars('policy'))
    ac.global_norm_policy = accum_gradient(grad, opt_policy)

    ac.step_1 = ph.rewards[:,r] + GAMMA*next_state_value
    ac.td_error = ac.step_1 - ac.q_value

    repl = gradient_override(ac.q_value, ac.td_error)
    grad = opt_td.compute_gradients(repl, scope_vars('value'))
    ac.global_norm_qvalue = accum_gradient(grad, opt_td)
    copy_vars = zip(
        # Include moving_mean/variance, which are not TRAINABLE_VARIABLES
        scope_vars('_value', tf.GraphKeys.GLOBAL_VARIABLES),
        scope_vars('value', tf.GraphKeys.GLOBAL_VARIABLES)) + zip(
        scope_vars('_policy', tf.GraphKeys.GLOBAL_VARIABLES),
        scope_vars('policy', tf.GraphKeys.GLOBAL_VARIABLES))
    for t,w in copy_vars:
        ops.per_epoch.append(t.assign(FLAGS.tau*w + (1-FLAGS.tau)*t))

    stats.reward_sum.append(tf.reduce_sum(ph.rewards[:,r]))
    stats.td_error.append(tf.reduce_sum(ac.td_error**2))

for r in range(REWARDS_ALL):
    with tf.name_scope('ac_%i' %r): make_acrl()

state = Struct(frames=np.zeros([STATE_FRAMES] + FRAME_DIM),
               count=0, last_obs=None, last_pos_reward=0,
               done=True, next_reset=False, last_reset=0)

envu.isRender = True # pybullet-gym
env.reset(); env.render()# Gym needs at least 1 reset&render before valid observation
def setup_key_actions():
    from pyglet.window import key
    a = np.array([0.]*max(3, ACTION_DIM))

    def settings_caption():
        d = dict(async=FLAGS.async,
            policy_index=app.policy_index,
            options=(['sample'] if FLAGS.sample_action else []) +
                (['quantize'] if FLAGS.quantize else []) +
                (['print'] if app.print_action else []))
        print(d)
        window.set_caption(str(d))

    def key_press(k, mod):
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
        if k==ord('e'):
            app.policy_index = -1
        elif k >= ord('1') and k <= ord('9'):
            app.policy_index = min(REWARDS_ALL-1, int(k - ord('1')))
        elif k==ord('a'):
            app.print_action ^= True
        elif k==ord('s'):
            FLAGS.sample_action ^= True
        elif k==ord('q'):
            FLAGS.quantize ^= True
        elif k==ord('i'):
            app.show_state_image = True
        elif k==ord('t'):
            training.enable ^= True
        elif k==ord('r'):
            state.next_reset = True
        else: return
        settings_caption()

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
    settings_caption()
    return a

action = Struct(to_take=None, policy=[], keyboard=setup_key_actions())
def step_to_frames():
    def softmax(x):
        e_x = np.exp(x - x.max())
        return e_x / e_x.sum()
    def choose_action(value): # Choose from Q-values or softmax policy
        return np.random.choice(value.shape[0], p=softmax(value)) if FLAGS.sample_action else np.argmax(value)

    a = action.keyboard[:ACTION_DIM].copy()
    if ACTION_DISCRETE: a = onehot_vector(int(a[0]+1.))

    if state.count > 0 and app.policy_index != -1:
        a = action.policy.copy()
        if 0:#not ACTION_DISCRETE:
            idx = choose_action(action.policy_value) if FLAGS.sample_action else 0
            a = ([action.policy] + EVAL_ACTIONS)[idx]
    if FLAGS.sample_action:
        np.random.seed(0)
        offset = np.array([math.sin(2*math.pi*(r + 8./(1+state.count))) for r in np.random.rand(ACTION_DIM)])
        a = np.clip(a+offset, -1, 1.)

    if FLAGS.quantize and QUANT_ACTIONS:
        idx = np.argmin(((np.array(QUANT_ACTIONS) - a)**2).sum(1))
        a = QUANT_ACTIONS[idx]

    env_action = a
    if ACTION_DISCRETE:
        env_action = np.argmax(a)
        a = onehot_vector(env_action)
    action.to_save = a
    if app.print_action: print(list(action.to_save), list(action.policy))#, list(action.policy_value))

    obs = state.last_obs
    reward_sum = 0.
    state.frames[:-1] = state.frames[1:]
    for frame in range(STATE_FRAMES):
        state.done |= state.next_reset
        state.last_reset += 1
        if state.done:
            state.next_reset = False
            state.last_reset = 0
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
            if state.last_reset > 100 and state.last_pos_reward > 40:
                state.done = True # Reset track if spinning
                reward = -100
        state.last_pos_reward = 0 if reward>0. or state.done else state.last_pos_reward+1
        reward_sum += reward
    state.last_obs = obs
    return [reward_sum]

ops.ph_step = [frame_to_state] + [
    i for sublist in [
        [ac.ph_policy[0],# Policy from uploaded state,
        ac.ph_policy_value[0],
        ] for ac in allac]
    for i in sublist]
with tf.get_default_graph().control_dependencies(ops.ph_step):
    ops.ph_step.append(tf.group(*ops.post_step))
def append_to_batch():
    save_paths = batch_paths(training.append_batch)
    if not FLAGS.async and FLAGS.recreate_states:
        if not state.count:
            training.saved_batch = mmap_batch(save_paths, 'r')
        batch = training.saved_batch
        state.frames = batch.rawframes[state.count]
        save_reward = batch.rewards[state.count]
        save_action = batch.actions[state.count]
    else:
        save_reward = step_to_frames()
        save_action = action.to_save

    ret = sess.run(ops.ph_step, feed_dict={ph.frame: [state.frames]})
    save_state = ret[0]
    action.policy = ret[1]
    action.policy_value = ret[2]
    if app.show_state_image:
        app.show_state_image = False
        proc = multiprocessing.Process(target=imshow, args=([save_state[0,:,:,i] for i in range(STATE_FRAMES)],))
        proc.start()

    temp_paths = batch_paths(FLAGS.async, 'temp')
    if not training.temp_batch:
        training.temp_batch = mmap_batch(temp_paths, 'w+')
    batch = training.temp_batch
    batch.rawframes[state.count] = state.frames
    batch.states[state.count] = save_state[0]
    batch.rewards[state.count] = save_reward
    batch.actions[state.count] = save_action

    state.count += 1
    if state.count == ER_BATCH_SIZE:
        if FLAGS.async or training.batches_recorded < FLAGS.batch_keep:
            print('Replacing batch #%i' % training.append_batch)
            for a in batch.arrays: del a
            training.temp_batch = None

            # Rename async batch files into server's ER batches.
            for k in save_paths.keys():
                src = temp_paths[k]
                dst = save_paths[k]
                os.system('rm -f ' + dst)
                os.system('mv ' + src + ' ' + dst)

        training.batches_recorded += 1
        training.append_batch += 1
        if FLAGS.async and training.batches_recorded == FLAGS.batch_per_async:
            training.batches_recorded = 0
            training.append_batch = FIRST_BATCH
        state.count = 0

if not FLAGS.async:
    ops.per_epoch = [
        accum_value(tf.concat([stats.td_error], 0)),
        tf.concat([[ac.global_norm_qvalue] for ac in allac], 0),
        tf.concat([[ac.global_norm_policy] for ac in allac], 0),
    ] + ops.per_epoch
    init_vars()

    ops.per_batch += tf.get_collection(tf.GraphKeys.UPDATE_OPS) # batch_norm
    with tf.get_default_graph().control_dependencies(ops.per_batch):
        ops.per_batch.append(tf.group(*ops.post_batch))
    for s in sorted(stats.__dict__.keys()):
        ops.per_batch.append(tf.concat([[r] for r in stats.__dict__[s]], 0))

    if FLAGS.summary:
        merged = tf.summary.merge_all()
        #ops.per_epoch += [merged]
        train_writer = tf.summary.FileWriter('/tmp/tf', sess.graph)

manager = multiprocessing.Manager()
batch_sets = manager.list()
def proc_batch_set():
    batch_set = {}
    while len(batch_set) < FLAGS.minibatch:
        try:
            idx = -np.random.choice(FLAGS.batch_keep+1) if np.random.choice(2) \
                else np.random.choice(FLAGS.batch_async)
            if idx in batch_set: continue
            b = mmap_batch(batch_paths(idx), 'r', rawframes=False)
            batch_set[idx] = b
        except Exception as e:
            print(e)
            continue
    batch_sets.append(batch_set.values())

proclist = []
mb = Struct(
    states = np.zeros([FLAGS.minibatch] + STATE_DIM),
    next_states = np.zeros([FLAGS.minibatch] + STATE_DIM),
    actions=np.zeros([FLAGS.minibatch, ACTION_DIM]),
    rewards=np.zeros([FLAGS.minibatch, REWARDS_GLOBAL]))
def make_minibatch(): # Each minibatch is random subset of batch trajectories
    if make_minibatch.current_step == 0:
        if 1:
            PROCESSES = 2
            while len(proclist) < PROCESSES:
                proc = multiprocessing.Process(target=proc_batch_set)
                proc.start()
                proclist.insert(0, proc)
            proc = proclist.pop()
            proc.join()
        else:
            proc_batch_set()
        make_minibatch.batch_set = batch_sets.pop()
        sess.run(ops.new_trajectories)

    step = make_minibatch.current_step
    next_step = step + 1
    if next_step == ER_BATCH_STEPS:
        next_step = 0
    make_minibatch.current_step = next_step

    for b,batch in enumerate(make_minibatch.batch_set):
        mb.states[b] = batch.states[step]
        mb.next_states[b] = batch.states[step+1]
        mb.actions[b] = batch.actions[step+1]
        mb.rewards[b] = batch.rewards[step+1]
make_minibatch.current_step = 0

def train_accum_minibatch():
    # Upload & train minibatch
    make_minibatch()
    feed_dict = {
        ph.states: mb.states,
        ph.next_states: mb.next_states,
        ph.actions: mb.actions,
        ph.rewards: mb.rewards}
    # imshow([mb.states[0][:,:,i] for i in range(STATE_FRAMES)])
    r = sess.run(ops.per_batch, feed_dict)
    if app.policy_index != -1:
        for i,s in enumerate(reversed(sorted(stats.__dict__.keys()))):
            sys.stdout.write(s+': ' + ('\n'if s[0]=='_'else'') + str(r[-i-1][app.policy_index])+'  ')
        print('')

def train_apply_gradients():
    r = sess.run(ops.per_epoch, feed_dict={ph.learning_rate: FLAGS.learning_rate})
    #w = scope_vars('ac_0/_value/dense', tf.GraphKeys.GLOBAL_VARIABLES)[3]; print(w, w.eval()[:10])
    app.epoch_count += 1
    sess.run(ops.post_epoch)
    pprint(dict(
        policy_index=app.policy_index,
        epoch_count=app.epoch_count,
        epoch_td_error=    r[0][app.policy_index],
        global_norm_qvalue=r[1][app.policy_index],
        global_norm_policy=r[2][app.policy_index],
        learning_rate='%.6f' % FLAGS.learning_rate,
        batches=dict(keep=FLAGS.batch_keep,async=FLAGS.batch_async,minibatch=FLAGS.minibatch)))
    os.system('clear') # Scroll up to see status

    if 0:#FLAGS.summary:
        summary = r[-1]
        train_writer.add_summary(summary, app.epoch_count)

def rl_loop():
    while not app.quit:
        if FLAGS.async or not training.enable or training.batches_recorded < FLAGS.batch_keep:
            append_to_batch()
            continue
        for i in range(1):
            train_accum_minibatch()
        train_apply_gradients()
        env.render() # Render needed for keyboard events
rl_loop()
