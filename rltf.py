import tensorflow as tf, numpy as np
from tensorflow.contrib import rnn
import sys, os, multiprocessing, time, math
from utils import *
from mhdpa import *
from pprint import pprint

# python rltf.py --batch_keep 6 --batch_queue 20 --record 0
# for i in {1..4}; do python rltf.py --inst $i --batch_per_inst 5 & sleep 1; done

flags = tf.app.flags
flags.DEFINE_integer('inst', 0, 'ID of agent that accumulates gradients on server')
flags.DEFINE_integer('batch_keep', 0, 'Batches recorded from user actions')
flags.DEFINE_integer('batch_queue', 200, 'Batches in queue recorded from agents')
flags.DEFINE_integer('batch_per_inst', 100, 'Batches recorded per agent instance')
flags.DEFINE_boolean('replay', False, 'Replay actions recorded in memmap array')
flags.DEFINE_boolean('recreate_states', False, 'Recreate kept states from saved raw frames')
flags.DEFINE_boolean('record', False, 'Record over kept batches')
flags.DEFINE_string('summary', '/tmp/tf', 'Summaries path for Tensorboard')
flags.DEFINE_string('env_seed', '', 'Seed number for new environment')
flags.DEFINE_float('sample_action', 0., 'Sample actions, or use policy')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_float('tau', 1e-3, 'Target network update rate')
flags.DEFINE_float('gamma', 0.99, 'Discount rate')
flags.DEFINE_float('decay', 0.01, 'Q-network L2 weight decay')
flags.DEFINE_integer('minibatch', 64, 'Minibatch size')
flags.DEFINE_boolean('tdc', True, 'TDC instead of target networks')
flags.DEFINE_string('nsteps', '1,10,20,30', 'List of multi-step returns for Q-function')
FLAGS = flags.FLAGS

PORT, PROTOCOL = 'localhost:2222', 'grpc'
if not FLAGS.inst:
    server = tf.train.Server({'local': [PORT]}, protocol=PROTOCOL, start=True)
sess = tf.InteractiveSession(PROTOCOL+'://'+PORT)

STATE_FRAMES = 5    # Frames an action is repeated for, combined into a state

DISCRETE_ACTIONS = []
ENV_NAME = os.getenv('ENV')
if not ENV_NAME:
    raise Exception('Missing ENV environment variable')
if ENV_NAME == 'CarRacing-v0':
    import gym.envs.box2d
    car_racing = gym.envs.box2d.car_racing
    car_racing.WINDOW_W = 800 # Default is huge
    car_racing.WINDOW_H = 600
    DISCRETE_ACTIONS = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 1],
    ] if 1 else []
elif ENV_NAME == 'FlappyBird-v0':
    import gym_ple # [512, 288]
elif 'Bullet' in ENV_NAME:
    import pybullet_envs

import gym
env = gym.make(ENV_NAME)
env._max_episode_steps = None # Disable step limit
envu = env.unwrapped

from pyglet import gl
def draw_line(a, b, color=(1,1,1,1)):
    gl.glLineWidth(3)
    gl.glBegin(gl.GL_LINES)
    gl.glColor4f(*color)
    gl.glVertex3f(window.width*a[0], window.height*(1-a[1]), 0)
    gl.glVertex3f(window.width*b[0], window.height*(1-b[1]), 0)
    gl.glEnd()
def draw_attention():
    if not app.draw_attention or state.ph_attention is None:
        return
    s = state.ph_attention.shape[1:3]
    for head in range(3):
        color = onehot_vector(head, 3)
        for y1 in range(s[0]):
            for x1 in range(s[1]):
                for y2 in range(s[0]):
                    for x2 in range(s[1]):
                        f = state.ph_attention[0, y1,x1, y2,x2, head]
                        if f < 0.1: continue
                        draw_line(
                            ((x1+0.5)/s[1], (y1+0.5)/s[0]),
                            ((x2+0.5)/s[1], (y2+0.5)/s[0]),
                            color+[f])
def hook_swapbuffers():
    flip = window.flip
    def hook():
        draw_attention()
        flip()
    window.flip = hook

ACTION_DIMS = (env.action_space.shape or [env.action_space.n])[0]
ACTION_DISCRETE = not env.action_space.shape
def onehot_vector(idx, dims): return [1. if idx == i else 0. for i in range(dims)]
ACTION_DOUBLE = 1
if ACTION_DISCRETE:
    MULTI_ACTIONS = [onehot_vector(a, ACTION_DIMS) for a in range(ACTION_DIMS)]
else:
    MULTI_ACTIONS = DISCRETE_ACTIONS
    ACTION_DOUBLE = 2
MULTI_ACTIONS = tf.constant(MULTI_ACTIONS, DTYPE)
POLICY_SOFTMAX = ACTION_DISCRETE
FRAME_DIM = list(env.observation_space.shape)

CONV_NET = len(FRAME_DIM) == 3
STATE_DIM = FRAME_DIM[:]
if CONV_NET:
    FRAME_LCN = False
    GRAYSCALE = True
    if GRAYSCALE and STATE_DIM[-1] == 3:
        STATE_DIM[-1] = 1
    CHANNELS = STATE_DIM[-1]

    RESIZE = [84, 84]
    if RESIZE:
        STATE_DIM[:2] = RESIZE
STATE_DIM[-1] *= STATE_FRAMES

FIRST_BATCH = -FLAGS.batch_keep
LAST_BATCH = FLAGS.batch_queue
ER_BATCH_SIZE = 100
ER_BATCH_STEPS = ER_BATCH_SIZE-1

def batch_paths(batch_num, path=None):
    if not path:
        path = '%s_%i_%i' % (ENV_NAME, STATE_FRAMES, ACTION_DOUBLE)
    os.system('mkdir -p batches')
    path = 'batches/' + path + '_%i_%s.mmap'
    return {key: path % (batch_num, key) for key in ['rawframes', 'states', 'actions', 'rewards']}

REWARDS_GLOBAL = REWARDS_ALL = 1
def mmap_batch(paths, mode, only_actions=False, states=True, rawframes=True):
    batch = Struct(actions=np.memmap(paths['actions'], DTYPE.name, mode, shape=(ER_BATCH_SIZE, ACTION_DIMS*ACTION_DOUBLE)))
    if only_actions:
        return batch.actions
    batch.rewards = np.memmap(paths['rewards'], DTYPE.name, mode, shape=(ER_BATCH_SIZE, REWARDS_GLOBAL))
    batch.arrays = [batch.actions, batch.rewards]
    if states:
        batch.states = np.memmap(paths['states'], DTYPE.name, mode, shape=(ER_BATCH_SIZE,) + tuple(STATE_DIM))
        batch.arrays.append(batch.states)
    if rawframes:
        batch.rawframes = np.memmap(paths['rawframes'], DTYPE.name, mode, shape=(ER_BATCH_SIZE, STATE_FRAMES) + tuple(FRAME_DIM))
        batch.arrays.append(batch.rawframes)
    return batch

training = Struct(enable=True, batches_recorded=0, batches_mtime={}, temp_batch=None,
    adv=True, ddpg=True,
    nsteps=[int(s) for s in FLAGS.nsteps.split(',') if s])
ph = Struct(
    adv=tf.placeholder('bool', ()),
    ddpg=tf.placeholder('bool', ()),
    states=tf.placeholder(DTYPE, [FLAGS.minibatch] + STATE_DIM),
    actions=tf.placeholder(DTYPE, [FLAGS.minibatch, ACTION_DIMS*ACTION_DOUBLE]),
    next_states=[tf.placeholder(DTYPE, [FLAGS.minibatch] + STATE_DIM) for n in training.nsteps],
    rewards=[tf.placeholder(DTYPE, [FLAGS.minibatch, REWARDS_GLOBAL]) for n in training.nsteps],
    frame=tf.placeholder(DTYPE, [1, STATE_FRAMES] + FRAME_DIM),
    nsteps=tf.placeholder('int32', [len(training.nsteps)]))

if CONV_NET:
    frame_to_state = tf.reshape(ph.frame, [-1] + FRAME_DIM) # Move STATE_FRAMES into batches
    if RESIZE:
        frame_to_state = tf.image.resize_images(frame_to_state, RESIZE, tf.image.ResizeMethod.AREA)
    if FRAME_LCN:
        frame_to_state = local_contrast_norm(frame_to_state, GAUSS_W)
        frame_to_state = tf.reduce_max(frame_to_state, axis=-1)
    else:
        if GRAYSCALE: frame_to_state = tf.reduce_mean(frame_to_state, axis=-1, keep_dims=True)
        frame_to_state = frame_to_state/255.
    frame_to_state = tf.transpose(frame_to_state, [1, 2, 0, 3])# Move STATE_FRAMES into channels
    frame_to_state = tf.reshape(frame_to_state, [1] + STATE_DIM)
else:
    frame_to_state = tf.reshape(ph.frame, [1] + STATE_DIM)

app = Struct(policy_index=0, quit=False, update_count=0, print_action=False, show_state_image=False, draw_attention=True)
if FLAGS.inst:
    FIRST_BATCH = (FLAGS.inst-1)*FLAGS.batch_per_inst
else:
    def init_vars(): sess.run(tf.global_variables_initializer())
    if FLAGS.record or FLAGS.replay or FLAGS.recreate_states:
        # Record new arrays
        app.policy_index = -1
    else:
        training.batches_recorded = FLAGS.batch_keep
training.append_batch = FIRST_BATCH

ops = Struct(per_minibatch=[], post_minibatch=[], per_update=[], post_update=[], post_step=[], new_batches=[])
def accum_value(value):
    accum = tf.Variable(tf.zeros_like(value), trainable=False)
    ops.per_minibatch.append(accum.assign_add(value))
    ops.post_update.append(accum.assign(tf.zeros_like(value)))
    return accum
def accum_gradient(grads, opt, clip_norm=10.):
    global_norm = None
    grads, weights = zip(*grads)
    grads = [accum_value(g) for g in grads]
    # Clip gradients by global norm to prevent destabilizing policy
    grads,global_norm = tf.clip_by_global_norm(grads, clip_norm)
    grads = zip(grads, weights)
    ops.per_update.append(opt.apply_gradients(grads))
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
    norm = tf.layers.BatchNormalization(_scope=_scope, scale=False, center=False)
    x = [norm.apply(x[i], training=i==layer_batch_norm.training_idx) for i in range(len(x))]
    for w in norm.weights: variable_summaries(w)
    return x

def layer_dense(x, outputs, activation=tf.nn.relu, norm=True, use_bias=False, trainable=True):
    if norm: x = layer_batch_norm(x)
    _scope = tf.contrib.framework.get_name_scope()
    dense = tf.layers.Dense(outputs, activation, use_bias, trainable=trainable, _scope=_scope)
    x = [dense.apply(n) for n in x]
    for w in dense.weights: variable_summaries(w)
    return x

def layer_lstm(x, ac):
    cell = rnn.LSTMCell(HIDDEN_NODES, use_peepholes=True)
    x = layer_batch_norm(x)
    with tf.name_scope('lstm'):
        scope = tf.contrib.framework.get_name_scope()
        for i in range(len(x)):
            create_initial_state = i<2
            is_inst = i==0
            if create_initial_state:
                with tf.name_scope(str(FLAGS.inst)):
                    batch_size = x[i].shape[0]
                    vars = [tf.Variable(s, trainable=False, collections=[None]) for s in cell.zero_state(batch_size, DTYPE)]
                    sess.run(tf.variables_initializer(vars))
                    initial_state = tf.contrib.rnn.LSTMStateTuple(*vars)
            else:
                initial_state = final_state

            x[i] = tf.expand_dims(x[i], 1) # [batch_size, max_time (i.e. 1), ...],
            output, final_state = tf.nn.dynamic_rnn(cell, x[i], initial_state=initial_state, scope=scope)
            x[i] = output[:,0,:]

            if create_initial_state:
                [ops.post_minibatch, ops.post_step][is_inst] += [initial_state[c].assign(final_state[c]) for c in range(2)]
                if not is_inst:
                    ops.new_batches += [initial_state[c].assign(tf.zeros_like(initial_state[c])) for c in range(2)]

        for w in cell.weights: variable_summaries(w)
    return x

def make_conv_net(x):
    LAYERS = [
        (32, 8, 4),
        (32, 4, 2),
        (32, 3, 1)]
    for l,(filters, width, stride) in enumerate(LAYERS):
        with tf.name_scope('conv'):
            x = layer_batch_norm(x)
            _scope = tf.contrib.framework.get_name_scope()
            conv = tf.layers.Conv2D(filters, width, stride, use_bias=False, activation=tf.nn.elu, _scope=_scope)
            x = [conv.apply(n) for n in x]
            for w in conv.weights: variable_summaries(w)
        print(x[0].shape)
    return x

def make_attention_net(x, ac):
    relational = MHDPA()
    for l in range(MHDPA_LAYERS):
        with tf.name_scope('mhdpa'):
            if l == 0:
                x = layer_batch_norm(x)
                x = [concat_coord_xy(n) for n in x]
            for i,n in enumerate(x):
                x[i], attention = relational.apply(n)
                if i==0 and l==(MHDPA_LAYERS-1):
                    ac.ph_attention = attention # Display agent attention
    return x

def make_shared(x, ac):
    print(x[0].shape)
    if CONV_NET:
        x = make_conv_net(x)
    if MHDPA_LAYERS:
        x = make_attention_net(x, ac)

    x = [tf.layers.flatten(n) for n in x]
    print(x[0].shape)
    return x

SHARED_LAYERS = True
MHDPA_LAYERS = 0#3
HIDDEN_NODES = 500

opt = Struct(td=tf.train.AdamOptimizer(FLAGS.learning_rate),
    policy=tf.train.AdamOptimizer(FLAGS.learning_rate/20),
    error=tf.train.AdamOptimizer(1))

allac = []
def make_acrl():
    ac = Struct(per_minibatch=Struct(), per_update=Struct())
    allac.append(ac)

    def make_dense(x, nodes=[HIDDEN_NODES]*2):
        for n in nodes:
            with tf.name_scope('hidden'):
                x = layer_dense(x, n)
        return x

    def make_policy(shared):
        hidden = make_dense(shared)
        with tf.name_scope('output'):
            softmax_dims = ACTION_DIMS if POLICY_SOFTMAX else len(DISCRETE_ACTIONS)
            if softmax_dims:
                policy = layer_dense(hidden, softmax_dims, tf.nn.softmax)
                if DISCRETE_ACTIONS:
                    policy = [tf.matmul(n,MULTI_ACTIONS) for n in policy]
            else:
                policy = layer_dense(hidden, ACTION_DIMS, tf.tanh)
        if ACTION_DOUBLE == 2:
            with tf.name_scope('multiply'):
                policy = [tf.concat([p,m],-1) for p,m in zip(policy, layer_dense(hidden, ACTION_DIMS, tf.nn.sigmoid))]
        return policy

    layer_batch_norm.training_idx = 1
    state_inputs = [frame_to_state] + ([] if FLAGS.inst else [ph.states]+ph.next_states)
    with tf.name_scope('policy'):
        shared = make_shared(state_inputs, ac)
        shared_weights = scope_vars() if SHARED_LAYERS else []
        policy = make_policy(shared)
        ac.ph_policy = policy[0]
        if not FLAGS.inst:
            ac.policy = policy[1]
            ac.policy_next = policy[2:]

    def multi_actions_pre(state, actions, idx, batch_size=FLAGS.minibatch, include_policy=True):
        num_actions = len(DISCRETE_ACTIONS)
        if not num_actions: return
        a = tf.tile(tf.expand_dims(MULTI_ACTIONS, 0), [batch_size, 1, ACTION_DOUBLE])
        if include_policy:
            num_actions += 1
            policy_action = tf.expand_dims(actions[idx], 1)
            a = tf.concat([policy_action, a], 1)
        action_combis = a
        n = tf.expand_dims(state[idx], 1)
        n = tf.tile(n, [1, num_actions, 1])
        n = tf.reshape(n, [batch_size*num_actions, -1])
        a = tf.reshape(a, [batch_size*num_actions, -1])
        state[idx], actions[idx] = n, a
    def multi_actions_post(n, batch_size=FLAGS.minibatch, reduce_max=False):
        num_actions = len(DISCRETE_ACTIONS)
        if not num_actions: return
        n = tf.reshape(n, [batch_size, int(n.shape[0])/batch_size])
        n = tf.reduce_max(n, 1) if reduce_max else n
        return n

    def make_value_output(state, actions):
        actions = [tf.concat([a, 1-a, a+1], -1) for a in actions]
        offsets = [0.05, 0.1, 0.2, 0.4]
        actions = [tf.concat([a] +
            [a+o for o in offsets] +
            [a-o for o in offsets], -1) for a in actions]

        # Prevent dead neurons if taking one-sided input
        double_relu = lambda x: tf.nn.relu(tf.concat([x, -x], -1))

        with tf.name_scope('actions'):
            if 1:
                a1 = layer_dense(actions, HIDDEN_NODES/2, double_relu, norm=False)
                combined = [n*a for n,a in zip(state, a1)]
            else:
                state = layer_batch_norm(state) # Only norm the state, not actions
                combined = [tf.concat([s,a], -1) for s,a in zip(state, actions)]
                combined = layer_dense(combined, HIDDEN_NODES, norm=False)
        with tf.name_scope('output'):
            output = layer_dense(combined, 1, None, norm=False)
            return [n[:,0] for n in output], combined

    gamma_nsteps = tf.expand_dims(tf.stack([FLAGS.gamma**tf.cast(ph.nsteps[i], DTYPE)
        for i in range(len(training.nsteps))]), 0)
    def make_qvalue(shared):
        if not SHARED_LAYERS: shared = make_shared(state_inputs, ac)
        with tf.name_scope('value'):
            state = make_dense(shared)
            if FLAGS.inst:
                actions = [ac.ph_policy]
            else:
                state = [state[0], state[1], state[1]] + state[2:]
                actions = [ac.ph_policy, ph.actions, ac.policy] + ac.policy_next
            multi_actions_pre(state, actions, 0, 1)

            q, combined = make_value_output(state, actions)
            # Q for all actions in agent's current state
            q[0] = multi_actions_post(q[0], 1)

            value = Struct(ph_policy=q[0])
            if FLAGS.inst: return value, None
            value.q = q[1]
            value.state = q[2]
            next_state_value = q[3:]

            value_weights = scope_vars() + shared_weights
        with tf.name_scope('error_value'):
            error_predict = layer_dense(combined, 1, None, norm=False)[1][:,0]
            error_weights = scope_vars()

        value.nstep = []
        for i in range(len(training.nsteps)):
            n_return = ph.rewards[i][:,r] + next_state_value[i] * gamma_nsteps[:,i]
            value.nstep.append(n_return)

        def post_common_target(common_target_value):
            td_error = common_target_value - value.q
            repl = gradient_override(error_predict, td_error-error_predict)
            grads = opt.error.compute_gradients(repl, error_weights)
            ac.per_update.gnorm_error = accum_gradient(grads, opt.error)

            repl = gradient_override(value.q, td_error)
            grad_s = opt.td.compute_gradients(repl, value_weights)
            # TDC averaged over all n-steps
            tdc_steps = len(training.nsteps)
            for i in range(tdc_steps):
                repl = gradient_override(next_state_value[i], -gamma_nsteps[:,i]*error_predict / tdc_steps)
                grad_s2 = opt.td.compute_gradients(repl, value_weights)
                for i in range(len(grad_s)):
                    (g, w), g2 = grad_s[i], grad_s2[i][0]
                    grad_s[i] = (g+g2, w)

            ac.td_error = td_error
            ac.per_update.gnorm_qvalue = accum_gradient(grad_s, opt.td)

        return value, post_common_target

    [value1, fn1] = make_qvalue(shared)
    ac.value1 = value1

    if FLAGS.inst: return
    [value2, fn2] = make_qvalue(shared)

    target_value = -sys.float_info.max
    for i in range(len(training.nsteps)):
        # Fix value overestimation by clipping the actor's critic with a second critic,
        # to avoid the bias introduced by the policy update.
        nstep_return = tf.minimum(value1.nstep[i], value2.nstep[i])
        # Maximize over the n-step returns
        target_value = tf.maximum(target_value, nstep_return)
    fn1(target_value)
    fn2(target_value)

    policy_weights = scope_vars('policy')
    if 1: # DDPG-style policy updates using Q-function gradient
        value_grad = tf.gradients(value1.state, ac.policy)[0]
        value_grad = tf.where(ph.ddpg, value_grad, tf.zeros_like(value_grad))
        repl = gradient_override(ac.policy, value_grad)
        grads = opt.policy.compute_gradients(repl, policy_weights)
        ac.per_update.gnorm_policy_ddpg = accum_gradient(grads, opt.policy)

    if 1: # Policy updates using multi-step advantage value
        adv = tf.maximum(0., target_value - value1.state) # Only towards better actions
        adv = tf.where(ph.adv, adv, tf.zeros_like(adv))
        adv = tf.expand_dims(adv, -1)
        a_diff = ph.actions - ac.policy
        repl = gradient_override(ac.policy, adv*a_diff)
        grads_adv = opt.policy.compute_gradients(repl, policy_weights)
        ac.per_update.gnorm_policy_adv = accum_gradient(grads_adv, opt.policy)

    if 0:
        for i in [-1]: # Q-network weight decay
            g,w = grad_s[i]
            grad_s[i] = (g + FLAGS.decay*w, w)
        copy_vars = zip(
            # Include moving_mean/variance, which are not TRAINABLE_VARIABLES
            scope_vars('_value', tf.GraphKeys.GLOBAL_VARIABLES),
            scope_vars('value', tf.GraphKeys.GLOBAL_VARIABLES)) + zip(
            scope_vars('_policy', tf.GraphKeys.GLOBAL_VARIABLES),
            scope_vars('policy', tf.GraphKeys.GLOBAL_VARIABLES))
        for t,w in copy_vars:
            ops.per_update.append(t.assign(FLAGS.tau*w + (1-FLAGS.tau)*t))

    ac.per_minibatch.reward_sum = tf.reduce_sum(ph.rewards[0][:,r])
    ac.per_minibatch.td_error = tf.reduce_sum(ac.td_error**2)
    ac.per_minibatch.policy_min = tf.reduce_min(ac.policy, axis=0)
    ac.per_minibatch.policy_max = tf.reduce_max(ac.policy, axis=0)

for r in range(REWARDS_ALL):
    with tf.name_scope('ac'): make_acrl()

state = Struct(frames=np.zeros([STATE_FRAMES] + FRAME_DIM),
               count=0, last_obs=None, last_pos_reward=0,
               done=True, next_reset=False, last_reset=0,
               ph_attention=None)

envu.isRender = True # pybullet-gym
env.reset(); env.render()# Gym needs at least 1 reset&render before valid observation
def setup_key_actions():
    from pyglet.window import key
    a = np.array([0.]*max(3, ACTION_DIMS))

    def settings_caption():
        d = dict(inst=FLAGS.inst,
            policy_index=app.policy_index,
            options=(['sample '+str(FLAGS.sample_action)] if FLAGS.sample_action else []) +
                (['print'] if app.print_action else []) +
                (['attention'] if app.draw_attention else []))
        print(d)
        window.set_caption(str(d))

    def key_press(k, mod):
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
            FLAGS.sample_action = 0.
        elif k==ord('i'):
            app.show_state_image = True
        elif k==ord('v'):
            training.adv ^= True
        elif k==ord('g'):
            training.ddpg ^= True
        elif k==ord('t'):
            training.enable ^= True
        elif k==ord('k'):
            # Bootstrap learning with user-supplied trajectories, then turn them off
            FLAGS.batch_keep = 0
        elif k==ord('r'):
            state.next_reset = True
        elif k==ord('m'):
            app.draw_attention ^= True
        else: return
        settings_caption()

    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0

    if not hasattr(envu, 'viewer'): # pybullet-gym
        return a
    global window
    window = envu.viewer.window
    window.on_key_press = key_press
    window.on_key_release = key_release
    window.on_close = lambda: setattr(app, 'quit', True)
    settings_caption()
    hook_swapbuffers()
    return a

action = Struct(to_take=None, policy=[], keyboard=setup_key_actions())
def step_to_frames():
    def softmax(x):
        e_x = np.exp(x - x.max())
        return e_x / e_x.sum()
    def choose_action(value): # Choose from Q-values or softmax policy
        return np.random.choice(value.shape[0], p=softmax(value)) if FLAGS.sample_action else np.argmax(value)
    def interp(f, a, b): return a + f*(b-a)

    a = action.keyboard[:ACTION_DIMS].copy()
    if ACTION_DISCRETE: a = onehot_vector(int(a[0]+1., ACTION_DIMS))
    if ACTION_DOUBLE == 2: a = np.concatenate([a,a], 0)

    if state.count > 0 and app.policy_index != -1:
        a = action.policy.copy()
        if 0:#not ACTION_DISCRETE:
            idx = choose_action(action.policy_value) if FLAGS.sample_action else 0
            a = ([action.policy] + MULTI_ACTIONS)[idx]
    if FLAGS.sample_action:
        np.random.seed(0)
        offset = np.array([FLAGS.sample_action*math.sin(2*math.pi*(r + state.count/10.)) for r in np.random.rand(ACTION_DIMS*ACTION_DOUBLE)])
        a = np.clip(a+offset, -1, 1.)

    env_action = a
    if ACTION_DISCRETE:
        env_action = np.argmax(a)
        a = onehot_vector(env_action, ACTION_DIMS)
    action.to_save = a
    if app.print_action: print(list(action.to_save), list(action.policy))#, list(action.policy_value))

    obs = state.last_obs
    reward_sum = 0.
    state.frames[:-1] = state.frames[1:]
    for frame in range(STATE_FRAMES):
        state.done |= state.next_reset
        state.last_reset += 1
        if state.done:
            state.last_pos_reward = 0
            state.next_reset = False
            state.last_reset = 0
            # New episode
            if FLAGS.env_seed:
                env.seed(int(FLAGS.env_seed))
            obs = env.reset()
        env.render()
        #imshow([obs, test_lcn(obs, sess)[0]])
        state.frames[frame] = obs

        if ACTION_DOUBLE == 2:
            a,b = env_action[:ACTION_DIMS], env_action[ACTION_DIMS:]
            lerp = interp(frame/(STATE_FRAMES-1.), a, a*b)
        else: lerp = env_action

        obs, reward, state.done, info = env.step(lerp)
        state.last_pos_reward = 0 if reward>0. else state.last_pos_reward+1
        if ENV_NAME == 'MountainCar-v0':
            # Mountain car env doesnt give any +reward
            reward = 1. if state.done else 0.
        elif ENV_NAME == 'CarRacing-v0' and not FLAGS.record:
            if state.last_pos_reward > 100 or not any([len(w.tiles) for w in envu.car.wheels]):
                state.done = True # Reset track if on grass
                reward = -100
        reward_sum += reward
    state.last_obs = obs
    return [reward_sum]

ops.ph_step = [frame_to_state] + [
    i for sublist in [
        [ac.ph_policy[0],# Policy from uploaded state,
        ac.value1.ph_policy[0],
        ] + ([ac.ph_attention] if MHDPA_LAYERS else [])
        for ac in allac]
    for i in sublist]
with tf.get_default_graph().control_dependencies(ops.ph_step):
    ops.ph_step.append(tf.group(*ops.post_step))
def append_to_batch():
    save_paths = batch_paths(training.append_batch)
    if not FLAGS.inst and FLAGS.recreate_states:
        if not state.count:
            training.saved_batch = mmap_batch(save_paths, 'r', states=False)
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
    if MHDPA_LAYERS: state.ph_attention = ret[3]
    if app.show_state_image:
        app.show_state_image = False
        proc = multiprocessing.Process(target=imshow,
            args=([save_state[0,:,:,CHANNELS*i:CHANNELS*(i+1)] for i in range(STATE_FRAMES)],))
        proc.start()

    temp_paths = batch_paths(FLAGS.inst, 'temp')
    if not training.temp_batch:
        training.temp_batch = mmap_batch(temp_paths, 'w+')
    batch = training.temp_batch
    batch.rawframes[state.count] = state.frames
    batch.states[state.count] = save_state[0]
    batch.rewards[state.count] = save_reward
    batch.actions[state.count] = save_action

    state.count += 1
    if state.count == ER_BATCH_SIZE:
        if FLAGS.inst or training.batches_recorded < FLAGS.batch_keep:
            print('Replacing batch #%i' % training.append_batch)
            for a in batch.arrays: del a
            training.temp_batch = None

            # Rename inst batch files into server's ER batches.
            for k in save_paths.keys():
                src = temp_paths[k]
                dst = save_paths[k]
                os.system('rm -f ' + dst)
                os.system('mv ' + src + ' ' + dst)

        training.batches_recorded += 1
        training.append_batch += 1
        if FLAGS.inst and training.batches_recorded == FLAGS.batch_per_inst:
            training.batches_recorded = 0
            training.append_batch = FIRST_BATCH
        state.count = 0

def tensor_dict_print(r, dict_name):
    if app.policy_index == -1: return
    keys = sorted(allac[0].__dict__[dict_name].__dict__.keys())
    np.set_printoptions(suppress=True, precision=6, sign=' ')
    d = {s: str(r[-i-1][app.policy_index]) for i,s in enumerate(reversed(keys))}
    pprint({dict_name: d})
def tensor_dict_compile(dict_name):
    keys = sorted(allac[0].__dict__[dict_name].__dict__.keys())
    ops.__dict__[dict_name] += [tf.concat([[allac[r].__dict__[dict_name].__dict__[s]]
        for r in range(len(allac))], 0) for s in keys]

if not FLAGS.inst:
    init_vars()

    ops.per_minibatch += tf.get_collection(tf.GraphKeys.UPDATE_OPS) # batch_norm
    with tf.get_default_graph().control_dependencies(ops.per_minibatch):
        ops.per_minibatch.append(tf.group(*ops.post_minibatch))
    tensor_dict_compile('per_minibatch')

    tensor_dict_compile('per_update')
    if FLAGS.summary:
        train_writer = tf.summary.FileWriter(FLAGS.summary, sess.graph)
        merged = tf.summary.merge_all()
        ops.per_update.insert(0, merged)

manager = multiprocessing.Manager()
batch_sets = manager.list()
def proc_batch_set():
    batch_set = {}
    while len(batch_set) < FLAGS.minibatch:
        try:
            idx = -np.random.choice(FLAGS.batch_keep+1) if np.random.choice(2) \
                else np.random.choice(FLAGS.batch_queue)
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
    actions = np.zeros([FLAGS.minibatch, ACTION_DIMS*ACTION_DOUBLE]),
    next_states = [np.zeros([FLAGS.minibatch] + STATE_DIM) for n in training.nsteps],
    rewards = [np.zeros([FLAGS.minibatch, REWARDS_GLOBAL]) for n in training.nsteps],
    nsteps = range(len(training.nsteps)))
def make_minibatch(): # Each minibatch is random subset of batch trajectories
    next_step = make_minibatch.current_step + 1
    if next_step == ER_BATCH_STEPS:
        make_minibatch.current_step = 0
        next_step = 1

    step = make_minibatch.current_step
    make_minibatch.current_step = next_step
    if step == 0:
        sess.run(ops.new_batches)
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

    for b,batch in enumerate(make_minibatch.batch_set):
        mb.states[b] = batch.states[step]
        mb.actions[b] = batch.actions[next_step]
        for i,nsteps in enumerate(training.nsteps):
            accum_reward = 0.
            for n in range(nsteps):
                last_state = next_step + n
                accum_reward += batch.rewards[last_state] * FLAGS.gamma**n
                if last_state == ER_BATCH_SIZE-1:
                    break
            mb.rewards[i][b] = accum_reward
            mb.next_states[i][b] = batch.states[last_state]
    for i,n in enumerate(training.nsteps):
        mb.nsteps[i] = min(ER_BATCH_SIZE-next_step, n)
make_minibatch.current_step = 0

def train_accum_minibatch():
    # Upload & train minibatch
    make_minibatch()
    feed_dict = {
        ph.adv: training.adv,
        ph.ddpg: training.ddpg,
        ph.states: mb.states,
        ph.actions: mb.actions,
        ph.nsteps: mb.nsteps}
    for i in range(len(training.nsteps)):
        feed_dict[ph.next_states[i]] = mb.next_states[i]
        feed_dict[ph.rewards[i]] = mb.rewards[i]
    # imshow([mb.states[0][:,:,i] for i in range(STATE_FRAMES)])
    r = sess.run(ops.per_minibatch, feed_dict)
    tensor_dict_print(r, 'per_minibatch')

def train_apply_gradients():
    r = sess.run(ops.per_update, feed_dict={})
    tensor_dict_print(r, 'per_update')

    app.update_count += 1
    sess.run(ops.post_update)
    pprint(dict(
        policy_updates=dict(adv=training.adv, ddpg=training.ddpg),
        nsteps=mb.nsteps,
        update_count=app.update_count,
        rates=dict(learning_rate=FLAGS.learning_rate, tau=FLAGS.tau, decay=FLAGS.decay, gamma=FLAGS.gamma),
        batches=dict(keep=FLAGS.batch_keep,inst=FLAGS.batch_queue,minibatch=FLAGS.minibatch)))
    os.system('clear') # Scroll up to see status

    if FLAGS.summary and not app.update_count%100:
        summary = r[0]
        train_writer.add_summary(summary, app.update_count)

def rl_loop():
    while not app.quit:
        if FLAGS.inst or not training.enable or training.batches_recorded < FLAGS.batch_keep:
            append_to_batch()
            continue
        for i in range(1):
            train_accum_minibatch()
        train_apply_gradients()
        env.render() # Render needed for keyboard events
rl_loop()
