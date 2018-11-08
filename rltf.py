import tensorflow as tf, numpy as np
from tensorflow.contrib import rnn
import sys, os, multiprocessing, time, math
from utils import *
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
flags.DEFINE_integer('minibatch', 64, 'Minibatch size')
flags.DEFINE_boolean('tdc', True, 'TDC instead of target networks')
flags.DEFINE_string('nsteps', '1,30', 'List of multi-step returns for Q-function')
FLAGS = flags.FLAGS

PORT, PROTOCOL = 'localhost:2222', 'grpc'
if not FLAGS.inst:
    server = tf.train.Server({'local': [PORT]}, protocol=PROTOCOL, start=True)
sess = tf.InteractiveSession(PROTOCOL+'://'+PORT)

# Frames an action is repeated for, combined into a state
ACTION_REPEAT = 3
# Concat frame states instead of repeating actions over multiple frames
CONCAT_STATES = 1

USE_LSTM = False
SHARED_LAYERS = True
BATCH_NORM = True
MHDPA_LAYERS = 0#3
FC_UNITS = 500

DISCRETE_ACTIONS = []
ACTION_TANH = []
ENV_NAME = os.getenv('ENV')
if not ENV_NAME:
    raise Exception('Missing ENV environment variable')
if ENV_NAME == 'CarRacing-v0':
    import gym.envs.box2d
    car_racing = gym.envs.box2d.car_racing
    car_racing.WINDOW_W = 800 # Default is huge
    car_racing.WINDOW_H = 600
    ACTION_TANH = [0]
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
    if not app.draw_attention or state.inst_attention is None:
        return
    s = state.inst_attention.shape[1:3]
    for head in range(3):
        color = onehot_vector(head, 3)
        for y1 in range(s[0]):
            for x1 in range(s[1]):
                for y2 in range(s[0]):
                    for x2 in range(s[1]):
                        f = state.inst_attention[0, y1,x1, y2,x2, head]
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
FINITE_DIFF_Q = 0#not ACTION_DISCRETE
if ACTION_DISCRETE:
    MULTI_ACTIONS = [onehot_vector(a, ACTION_DIMS) for a in range(ACTION_DIMS)]
else:
    MULTI_ACTIONS = []
    if ACTION_REPEAT >= 5:
        # Interpolate actions if repeating for many frames
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
STATE_DIM[-1] *= ACTION_REPEAT

FIRST_BATCH = -FLAGS.batch_keep
LAST_BATCH = FLAGS.batch_queue
ER_BATCH_SIZE = 100

def batch_paths(batch_num, path=None):
    if not path:
        path = '%s_%i_%i' % (ENV_NAME, ACTION_REPEAT, ACTION_DOUBLE)
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
        batch.rawframes = np.memmap(paths['rawframes'], DTYPE.name, mode, shape=(ER_BATCH_SIZE, ACTION_REPEAT) + tuple(FRAME_DIM))
        batch.arrays.append(batch.rawframes)
    return batch

training = Struct(enable=True, batches_recorded=0, batches_mtime={}, temp_batch=None,
    adv=True, ddpg=True,
    nsteps=[int(s) for s in FLAGS.nsteps.split(',') if s])
ph = Struct(
    adv=tf.placeholder('bool', ()),
    ddpg=tf.placeholder('bool', ()),
    actions=tf.placeholder(DTYPE, [FLAGS.minibatch, ACTION_DIMS*ACTION_DOUBLE]),
    states=[tf.placeholder(DTYPE, [FLAGS.minibatch, CONCAT_STATES] + STATE_DIM) for n in training.nsteps+[1]],
    rewards=tf.placeholder(DTYPE, [FLAGS.minibatch, len(training.nsteps), REWARDS_GLOBAL]),
    frame=tf.placeholder(DTYPE, [CONCAT_STATES, ACTION_REPEAT] + FRAME_DIM),
    nsteps=tf.placeholder('int32', [len(training.nsteps)]),
    inst_explore=tf.placeholder(DTYPE, ()))

if CONV_NET:
    frame_to_state = tf.reshape(ph.frame, [-1] + FRAME_DIM) # Combine CONCAT_STATES and ACTION_REPEAT
    if RESIZE:
        frame_to_state = tf.image.resize_images(frame_to_state, RESIZE, tf.image.ResizeMethod.AREA)
    if FRAME_LCN:
        frame_to_state = local_contrast_norm(frame_to_state, GAUSS_W)
        frame_to_state = tf.reduce_max(frame_to_state, axis=-1)
    else:
        if GRAYSCALE: frame_to_state = tf.reduce_mean(frame_to_state, axis=-1, keep_dims=True)
        frame_to_state = frame_to_state/255.
    frame_to_state = tf.reshape(frame_to_state, [CONCAT_STATES, ACTION_REPEAT] + RESIZE)
    frame_to_state = tf.transpose(frame_to_state, [0, 2, 3, 1])# Move ACTION_REPEAT into channels
else:
    frame_to_state = tf.reshape(ph.frame, [CONCAT_STATES] + STATE_DIM)

app = Struct(policy_index=0, quit=False, update_count=0, print_action=False, show_state_image=False,
    draw_attention=False, wireframe=True)
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
def accum_gradient(grads, opt, clip_norm=1.):
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
    norm = tf.layers.BatchNormalization(scale=False, center=False, momentum=0.9)
    x = [norm.apply(x[i], training=i==layer_batch_norm.training_idx) for i in range(len(x))]
    for w in norm.weights: variable_summaries(w)
    return x

def layer_dense(x, outputs, activation=None, use_bias=False, trainable=True):
    dense = tf.layers.Dense(outputs, activation, use_bias, trainable=trainable)
    x = [dense.apply(n) for n in x]
    for w in dense.weights: variable_summaries(w)
    return x

def layer_lstm(x, outputs, ac):
    cell = rnn.LSTMCell(outputs, use_peepholes=True)
    for i in range(len(x)):
        next_state = i - (len(x) - len(training.nsteps))
        create_initial_state = i<2
        if create_initial_state:
            with tf.variable_scope(str(FLAGS.inst)):
                zero_state = cell.zero_state(x[i].shape[0], DTYPE)
                vars = [tf.Variable(s, trainable=False, collections=[None]) for s in zero_state]
                sess.run(tf.variables_initializer(vars))
                initial_state = tf.contrib.rnn.LSTMStateTuple(*vars)
        elif next_state >= 0: # Next-states
            initial_state = final_state
            nsteps = [0]+training.nsteps
            if (nsteps[next_state+1]-nsteps[next_state]) != 1:
                print('Non-contiguous nsteps=%s' % nsteps) # Might break LSTM context
                initial_state = zero_state

        x[i], final_state = cell.call(x[i], initial_state)
        for w in cell.weights: variable_summaries(w)

        if create_initial_state:
            [ops.post_step, ops.post_minibatch][i] += [initial_state[c].assign(final_state[c]) for c in range(2)]
            if i==1: ops.new_batches += [initial_state[c].assign(tf.zeros_like(initial_state[c])) for c in range(2)]
    return x

def make_conv_net(x):
    LAYERS = [
        (32, 8, 4, 0),
        (32, 4, 2, 0),
        (32, 3, 1, 0)]
    x = [tf.expand_dims(n,-1) for n in x]
    for l,(filters, width, stride, conv3d) in enumerate(LAYERS):
        with tf.variable_scope('conv_%i' % l):
            pool_stride = None
            if stride >= 2: pool_stride = [1,2,2,1]; stride /= 2 # Smoothen filter responses

            kwds = dict(activation=None, use_bias=False)
            if conv3d:
                width3d, stride3d = conv3d
                conv = tf.layers.Conv3D(filters, (width, width, width3d), (stride, stride, stride3d), **kwds)
            else:
                if len(x[0].shape) == 5: # Back to 2D conv
                    x = [tf.reshape(n, n.shape.as_list()[:3] + [-1]) for n in x]
                conv = tf.layers.Conv2D(filters, width, stride, **kwds)
            x = [conv.apply(n) for n in x]
            for w in conv.weights: variable_summaries(w)

            if pool_stride: x = [tf.nn.avg_pool(n, pool_stride, pool_stride, 'VALID') for n in x]
            if BATCH_NORM: x = layer_batch_norm(x)
            x = [tf.tanh(n) for n in x]
        print(x[0].shape)
    return x

def make_attention_net(x, ac):
    for l in range(MHDPA_LAYERS):
        with tf.variable_scope('mhdpa_%i' % l):
            x = layer_batch_norm(x)
            if l == 0:
                relational = MHDPA()
                x = [concat_coord_xy(n) for n in x]
            for i,n in enumerate(x):
                x[i], attention = relational.apply(n)
                if i==0 and l==(MHDPA_LAYERS-1):
                    ac.inst_attention = attention # Display agent attention
                if i==1: ac.per_minibatch.__dict__['attention_minmax_%i'%l] = tf.stack([tf.reduce_min(attention), tf.reduce_max(attention)])
    return x

opt = Struct(td=tf.train.AdamOptimizer(FLAGS.learning_rate),
    policy=tf.train.AdamOptimizer(FLAGS.learning_rate/20),
    error=tf.train.AdamOptimizer(1))

allac = []
def make_acrl():
    ac = Struct(per_minibatch=Struct(), per_update=Struct())
    allac.append(ac)

    # Move concat states to last dimension
    state_inputs = [tf.expand_dims(frame_to_state,0)] + ([] if FLAGS.inst else ph.states)
    idx = range(len(state_inputs[0].shape))
    idx = [0] + idx[2:]
    idx.insert(-1, 1)
    CONCAT_STATE_DIM = STATE_DIM[:]
    CONCAT_STATE_DIM[-1] *= CONCAT_STATES
    state_inputs = [tf.reshape(tf.transpose(n,idx), [-1]+CONCAT_STATE_DIM) for n in state_inputs]

    def make_shared():
        x = state_inputs
        print(x[0].shape)
        if CONV_NET: x = make_conv_net(x)
        if MHDPA_LAYERS: x = make_attention_net(x, ac)

        x = [tf.layers.flatten(n) for n in x]
        print(x[0].shape)
        return x

    def make_fc(x, actions):
        with tf.variable_scope('hidden_0'):
            x = layer_dense(x, FC_UNITS)
            if BATCH_NORM: x = layer_batch_norm(x)
            x = [tf.nn.relu(n) for n in x]
        with tf.variable_scope('hidden_1'):
            if actions: x = [tf.concat([n,a],-1) for n,a in zip(x,actions)]
            if USE_LSTM: x = layer_lstm(x, FC_UNITS, ac)
            else: x = layer_dense(x, FC_UNITS)
            x = [double_relu(n) for n in x]
        return x

    def make_policy_output(hidden):
        if POLICY_SOFTMAX:
            policy = layer_dense(hidden, ACTION_DIMS, tf.nn.softmax)
            return policy

        def binary_output(scope, dims=ACTION_DIMS, tanh=False):
            with tf.variable_scope(scope):
                logit = layer_dense(hidden, dims)
                logit[0] += ph.inst_explore * tf.random_normal(logit[0].shape)
                r = [tf.tanh(n) if tanh else tf.sigmoid(n) for n in logit]

            for i in [0]:#range(len(r)):
                bin = logit[i]
                bin = tf.where(bin>0., tf.ones_like(bin), tf.zeros_like(bin))
                bin = bin*2. - 1. if tanh else bin
                #if i==1: r.insert(2, bin) # Discontinuous version of policy
                #else: r[i] = bin
            return r

        policy = [m1*(.5+.5*m2) for m1,m2 in zip(binary_output('m1'), binary_output('m2'))]
        if ACTION_TANH: # Multiply specific axes with tanh
            tanh = binary_output('tanh', len(ACTION_TANH), True)
            for i in range(len(policy)):
                mult = [tf.ones_like(tanh[i][:,0])]*ACTION_DIMS
                for j,t in enumerate(ACTION_TANH):
                    mult[t] = tanh[i][:,j]
                policy[i] *= tf.stack(mult, -1)

        if ACTION_DOUBLE == 2:
            policy = [tf.concat([p,m],-1) for p,m in zip(policy, binary_output('interp'))]
        return policy

    layer_batch_norm.training_idx = 1
    with tf.variable_scope('policy'):
        shared = make_shared()
        shared_weights = scope_vars() if SHARED_LAYERS else []
        hidden = make_fc(shared, None)
        with tf.variable_scope('output'):
            policy = make_policy_output(hidden)
        ac.inst_policy = policy[0]
        if not FLAGS.inst:
            ac.policy = policy[1]
            ac.policy_next = policy[-len(training.nsteps):]

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
        if not num_actions: return n
        n = tf.reshape(n, [batch_size, int(n.shape[0])/batch_size])
        n = tf.reduce_max(n, 1) if reduce_max else n
        return n

    STEP_FUNCS = 10
    def tile_actions(actions):
        if ACTION_DISCRETE:
            return actions

        rng = range(1-STEP_FUNCS, STEP_FUNCS)
        actions = [tf.nn.relu(tf.concat([STEP_FUNCS*a + f +0.5 for f in rng], -1)) for a in actions]
        actions = [tf.where(a>=1, #tf.logical_and(a>=1., a<2),
            tf.ones_like(a), # Discontinuous step functions BETTER
            #a if i==2 else # Attempt at gradient for policy value
            tf.zeros_like(a)) for i,a in enumerate(actions)]
        actions = [tf.concat([a, 1-a], -1) for a in actions]
        return actions

    gamma_nsteps = tf.expand_dims(tf.stack([FLAGS.gamma**tf.cast(ph.nsteps[i], DTYPE)
        for i in range(len(training.nsteps))]), 0)
    def make_qvalue():
        state = shared if SHARED_LAYERS else make_shared()

        actions = [ac.inst_policy]
        if not FLAGS.inst:
            state = [state[0]] + [state[1]]*2 + state[2:]
            actions = [actions[0], ph.actions, ac.policy] + ac.policy_next
        multi_actions_pre(state, actions, 0, 1)

        combined = make_fc(state, tile_actions(actions))
        with tf.variable_scope('output'):
            q = [n[:,0] for n in layer_dense(combined, 1)]

        # Q for all actions in agent's current state
        q[0] = multi_actions_post(q[0], 1)
        value = Struct(inst_policy=q.pop(0))
        if FLAGS.inst: return value, None

        [value.q, value.state] = q[:2]
        value.state_h = []
        value.action_h = []
        next_state_value = q[-len(training.nsteps):]
        value.nstep = ph.rewards[:,:,r] + gamma_nsteps*tf.stack(next_state_value, -1)

        value_weights = scope_vars() + shared_weights
        with tf.variable_scope('error_value'):
            error_predict = layer_dense(combined, 1)[1][:,0]
            error_weights = scope_vars()

        def update_qvalue(common_target_value):
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
        return value, update_qvalue

    with tf.variable_scope('value1'):
        [value1, update1] = make_qvalue()
        ac.value1 = value1

    if FLAGS.inst: return
    with tf.variable_scope('value2'):
        [value2, update2] = make_qvalue()

    # Fix value overestimation by clipping the actor's critic with a second critic,
    # to avoid the bias introduced by the policy update.
    min_values = tf.minimum(value1.nstep, value2.nstep)
    # Maximize over the n-step returns
    return_value = tf.reduce_max(min_values, -1)

    update1(return_value)
    update2(return_value)

    policy_weights = scope_vars('policy')
    a_diff = ph.actions - ac.policy
    if FINITE_DIFF_Q:
        value_grad = 0.
        zeros = tf.zeros_like(ac.policy)
        for state_h, div in zip(value1.state_h+[value1.q],
                                value1.action_h+[a_diff]):
            no_div0 = tf.abs(div) > 1./STEP_FUNCS
            diff = tf.expand_dims(state_h - value1.state, -1)
            value_grad += tf.where(no_div0, diff / div, zeros)
    else:
        # DDPG-style policy updates using Q-function gradient
        value_grad = tf.gradients(value1.state, ac.policy)[0]
    if value_grad is not None:
        value_grad = tf.where(ph.ddpg, value_grad, tf.zeros_like(value_grad))
        repl = gradient_override(ac.policy, value_grad)
        grads = opt.policy.compute_gradients(repl, policy_weights)
        ac.per_update.gnorm_policy_ddpg = accum_gradient(grads, opt.policy)

    if 1: # Policy updates using multi-step advantage value
        state_value = tf.maximum(value1.state, value2.state)
        adv = tf.maximum(0., return_value - state_value) # Only towards better actions
        adv = tf.where(ph.adv, adv, tf.zeros_like(adv))
        adv = tf.expand_dims(adv, -1)
        repl = gradient_override(ac.policy, adv*a_diff)
        grads_adv = opt.policy.compute_gradients(repl, policy_weights)
        ac.per_update.gnorm_policy_adv = accum_gradient(grads_adv, opt.policy)

    if 0:
        copy_vars = zip(
            # Include moving_mean/variance, which are not TRAINABLE_VARIABLES
            scope_vars('_value', True),
            scope_vars('value', True)) + zip(
            scope_vars('_policy', True),
            scope_vars('policy', True))
        for t,w in copy_vars:
            ops.per_update.append(t.assign(FLAGS.tau*w + (1-FLAGS.tau)*t))

    ac.per_minibatch.reward_sum = tf.reduce_sum(ph.rewards[:,0,r])
    ac.per_minibatch.td_error = tf.reduce_sum(ac.td_error**2)
    ac.per_minibatch.policy_min = tf.reduce_min(ac.policy, axis=0)
    ac.per_minibatch.policy_max = tf.reduce_max(ac.policy, axis=0)

for r in range(REWARDS_ALL):
    with tf.variable_scope('ac'): make_acrl()

state = Struct(frames=np.zeros([CONCAT_STATES, ACTION_REPEAT] + FRAME_DIM),
               count=0, last_obs=None, last_pos_reward=0,
               done=True, next_reset=False, last_reset=0,
               ph_attention=None)

def env_render():
    lines = lambda l: gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if l else gl.GL_FILL)
    lines(app.wireframe)
    env.render()
    lines(False)

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

    on_close = lambda: setattr(app, 'quit', True)
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
        elif k==ord('w'):
            app.wireframe ^= True
        elif k==ord('q'): on_close()
        else: return
        settings_caption()

    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0

    envu.isRender = True
    if not hasattr(envu, 'viewer'): # pybullet-gym
        return a
    global window
    env.reset(); env_render() # Needed for viewer.window
    window = envu.viewer.window
    window.on_key_press = key_press
    window.on_key_release = key_release
    window.on_close = on_close
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
    if ACTION_DISCRETE: a = onehot_vector(int(a[0]+1.), ACTION_DIMS)
    if ACTION_DOUBLE == 2: a = np.concatenate([a,[1.]*ACTION_DIMS], 0)

    if state.count > 0 and app.policy_index != -1:
        a = action.policy.copy()
        if 0:#not ACTION_DISCRETE:
            idx = choose_action(action.policy_value) if FLAGS.sample_action else 0
            a = ([action.policy] + MULTI_ACTIONS)[idx]
    '''
    if FLAGS.sample_action:
        np.random.seed(0)
        offset = np.array([FLAGS.sample_action*math.sin(2*math.pi*(r + state.count/20.)) for r in np.random.rand(ACTION_DIMS*ACTION_DOUBLE)])
        a = np.clip(a+offset, -1, 1.)
    '''

    env_action = a
    if ACTION_DISCRETE:
        env_action = np.argmax(a)
        a = onehot_vector(env_action, ACTION_DIMS)
    action.to_save = a
    if app.print_action: print(list(action.to_save), list(action.policy))#, list(action.policy_value))

    obs = state.last_obs
    reward_sum = 0.
    state.frames[:-1] = state.frames[1:]
    for frame in range(ACTION_REPEAT):
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
        env_render()
        #imshow([obs, test_lcn(obs, sess)[0]])
        state.frames[-1, frame] = obs

        if ACTION_DOUBLE == 2:
            a,b = env_action[:ACTION_DIMS], env_action[ACTION_DIMS:]
            lerp = interp(frame/(ACTION_REPEAT-1.), a, a*b)
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

ops.inst_step = [frame_to_state] + [
    i for sublist in [
        [ac.inst_policy[0],# Policy from uploaded state,
        ac.value1.inst_policy[0],
        ] + ([ac.inst_attention] if MHDPA_LAYERS else [])
        for ac in allac]
    for i in sublist]
with tf.get_default_graph().control_dependencies(ops.inst_step):
    ops.inst_step.append(tf.group(*ops.post_step))
def append_to_batch():
    save_paths = batch_paths(training.append_batch)
    if not FLAGS.inst and FLAGS.recreate_states:
        if not state.count:
            training.saved_batch = mmap_batch(save_paths, 'r', states=False)
        batch = training.saved_batch
        state.frames[-1] = batch.rawframes[state.count]
        save_reward = batch.rewards[state.count]
        save_action = batch.actions[state.count]
    else:
        save_reward = step_to_frames()
        save_action = action.to_save

    ret = sess.run(ops.inst_step, feed_dict={ph.frame: state.frames, ph.inst_explore: FLAGS.sample_action})
    save_state = ret[0]
    action.policy = ret[1]
    action.policy_value = ret[2]
    if MHDPA_LAYERS: state.inst_attention = ret[3]
    if app.show_state_image:
        app.show_state_image = False
        proc = multiprocessing.Process(target=imshow,
            args=([save_state[0,:,:,CHANNELS*i:CHANNELS*(i+1)] for i in range(ACTION_REPEAT)],))
        proc.start()

    temp_paths = batch_paths(FLAGS.inst, 'temp')
    if not training.temp_batch:
        training.temp_batch = mmap_batch(temp_paths, 'w+')
    batch = training.temp_batch
    batch.rawframes[state.count] = state.frames[-1]
    batch.states[state.count] = save_state[-1]
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
    shuffled = list(range(-1,-(FLAGS.batch_keep+1),-1)) + list(range(FLAGS.batch_queue))
    np.random.shuffle(shuffled)
    while len(batch_set) < FLAGS.minibatch:
        try:
            if not len(shuffled): return
            idx = shuffled.pop()
            b = mmap_batch(batch_paths(idx), 'r', rawframes=False)
            batch_set[idx] = b
        except Exception as e:
            print(e)
            continue
    batch_sets.append(batch_set.values())

proclist = []
mb = Struct(
    actions = np.zeros([FLAGS.minibatch, ACTION_DIMS*ACTION_DOUBLE]),
    states = [np.zeros([FLAGS.minibatch, CONCAT_STATES] + STATE_DIM) for n in training.nsteps+[1]],
    rewards = np.zeros([FLAGS.minibatch, len(training.nsteps), REWARDS_GLOBAL]),
    nsteps = list(range(len(training.nsteps))))
def make_minibatch(): # Each minibatch is random subset of batch trajectories
    step = make_minibatch.current_step
    if step+CONCAT_STATES == ER_BATCH_SIZE-1:
        step = 0

    if step == 0:
        sess.run(ops.new_batches)
        if 1:
            PROCESSES = 2
            while len(proclist) < PROCESSES:
                proc = multiprocessing.Process(target=proc_batch_set)
                proc.start()
                proclist.append(proc)
            proc = next((proc for proc in proclist if not proc.is_alive()), None)
            if proc:
                proc.join()
                proclist.remove(proc)
        else:
            proc_batch_set()
        if not len(batch_sets): return False
        make_minibatch.batch_set = batch_sets.pop()
    make_minibatch.current_step = step+1

    step += CONCAT_STATES
    for b,batch in enumerate(make_minibatch.batch_set):
        mb.states[0][b] = batch.states[step-CONCAT_STATES:step]
        mb.actions[b] = batch.actions[step]
        for i,nsteps in enumerate(training.nsteps):
            accum_reward = 0.
            for n in range(nsteps):
                next_step = step + n
                accum_reward += batch.rewards[next_step] * FLAGS.gamma**n
                next_step += 1
                if next_step == ER_BATCH_SIZE:
                    break
            mb.rewards[b,i] = accum_reward
            mb.states[i+1][b] = batch.states[next_step-CONCAT_STATES:next_step]
            if b==0:
                mb.nsteps[i] = next_step - step
    return True
make_minibatch.current_step = 0

def train_accum_minibatch():
    # Upload & train minibatch
    feed_dict = {
        ph.adv: training.adv,
        ph.ddpg: training.ddpg,
        ph.actions: mb.actions,
        ph.rewards: mb.rewards,
        ph.nsteps: mb.nsteps}
    for i in range(len(training.nsteps)+1):
        feed_dict[ph.states[i]] = mb.states[i]
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
        rates=dict(learning_rate=FLAGS.learning_rate, tau=FLAGS.tau, gamma=FLAGS.gamma),
        batches=dict(keep=FLAGS.batch_keep,inst=FLAGS.batch_queue,minibatch=FLAGS.minibatch)))
    os.system('clear') # Scroll up to see status

    if FLAGS.summary and not app.update_count%100:
        summary = r[0]
        train_writer.add_summary(summary, app.update_count)

def rl_loop():
    if app.quit: return False
    if FLAGS.inst or not training.enable or training.batches_recorded < FLAGS.batch_keep:
        append_to_batch()
    else:
        if make_minibatch():
            train_accum_minibatch()
            train_apply_gradients()
        env_render() # Render needed for keyboard events
    return True
import utils; utils.loop_while(rl_loop)
