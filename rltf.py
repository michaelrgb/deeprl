import tensorflow as tf, numpy as np
from tensorflow.contrib import rnn
import sys, os, multiprocessing, time, math
from pprint import pprint
from utils import *
import er

flags = tf.app.flags
flags.DEFINE_integer('inst', 0, 'ID of agent that accumulates gradients on server')
flags.DEFINE_integer('seq_keep', 0, 'Sequences recorded from user actions')
flags.DEFINE_integer('seq_inst', 128, 'Sequences in queue recorded from agents')
flags.DEFINE_integer('seq_per_inst', 64, 'Sequences recorded per agent instance')
flags.DEFINE_integer('minibatch', 64, 'Minibatch size')
flags.DEFINE_boolean('replay', False, 'Replay actions recorded in memmap array')
flags.DEFINE_boolean('recreate_states', False, 'Recreate kept states from saved raw frames')
flags.DEFINE_boolean('record', False, 'Record over kept sequences')
flags.DEFINE_string('summary', '/tmp/tf', 'Summaries path for Tensorboard')
flags.DEFINE_string('env_seed', '', 'Seed number for new environment')
flags.DEFINE_float('sample_action', 0., 'Sample actions, or use modal policy')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_float('tau', 1e-3, 'Target network update rate')
flags.DEFINE_float('gamma', 0.99, 'Discount rate')
flags.DEFINE_string('nsteps', '1,30', 'List of multi-step returns for Q-function')
flags.DEFINE_boolean('ddpg', True, 'Policy updates using Q-function gradient')
flags.DEFINE_boolean('adv', True, 'Policy updates using multi-step advantage value')
FLAGS = flags.FLAGS

PORT, PROTOCOL = 'localhost:2222', 'grpc'
if not FLAGS.inst:
    server = tf.train.Server({'local': [PORT]}, protocol=PROTOCOL, start=True)
sess = tf.InteractiveSession(PROTOCOL+'://'+PORT)

USE_LSTM = False
SHARED_LAYERS = False
BATCH_NORM = True
MHDPA_LAYERS = 0#3
FC_UNITS = 500

DISCRETE_ACTIONS = []
ENV_NAME = os.getenv('ENV')
er.ENV_NAME = ENV_NAME
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
    if not app.draw_attention or not MHDPA_LAYERS:
        return
    s = app.per_inst.attention.shape[1:3]
    for head in range(3):
        color = onehot_vector(head, 3)
        for y1 in range(s[0]):
            for x1 in range(s[1]):
                for y2 in range(s[0]):
                    for x2 in range(s[1]):
                        f = app.per_inst.attention[0, y1,x1, y2,x2, head]
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
if ACTION_DISCRETE:
    MULTI_ACTIONS = [onehot_vector(a, ACTION_DIMS) for a in range(ACTION_DIMS)]
else:
    MULTI_ACTIONS = []
    ACTION_TANH = range(ACTION_DIMS)
    if ENV_NAME == 'CarRacing-v0':
        ACTION_TANH = [0]
    ACTION_CLIP = [-1. if i in ACTION_TANH else 0. for i in range(ACTION_DIMS)], [1.]*ACTION_DIMS
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
STATE_DIM[-1] *= er.ACTION_REPEAT

training = Struct(enable=True, seq_recorded=0, batches_mtime={}, temp_batch=None,
    nsteps=[int(s) for s in FLAGS.nsteps.split(',') if s])
ph = Struct(
    adv=tf.placeholder('bool', ()),
    ddpg=tf.placeholder('bool', ()),
    actions=tf.placeholder(DTYPE, [FLAGS.minibatch, ACTION_DIMS]),
    states=[tf.placeholder(DTYPE, [FLAGS.minibatch, er.CONCAT_STATES] + STATE_DIM) for n in training.nsteps+[1]],
    rewards=tf.placeholder(DTYPE, [FLAGS.minibatch, len(training.nsteps), er.ER_REWARDS]),
    frame=tf.placeholder(DTYPE, [er.CONCAT_STATES, er.ACTION_REPEAT] + FRAME_DIM),
    nsteps=tf.placeholder('int32', [FLAGS.minibatch, len(training.nsteps)]),
    inst_explore=tf.placeholder(DTYPE, [ACTION_DIMS]))

if CONV_NET:
    frame_to_state = tf.reshape(ph.frame, [-1] + FRAME_DIM) # Combine CONCAT_STATES and er.ACTION_REPEAT
    if RESIZE:
        frame_to_state = tf.image.resize_images(frame_to_state, RESIZE, tf.image.ResizeMethod.AREA)
    if FRAME_LCN:
        frame_to_state = local_contrast_norm(frame_to_state, GAUSS_W)
        frame_to_state = tf.reduce_max(frame_to_state, axis=-1)
    else:
        if GRAYSCALE: frame_to_state = tf.reduce_mean(frame_to_state, axis=-1, keep_dims=True)
        frame_to_state = frame_to_state/255.
    frame_to_state = tf.reshape(frame_to_state, [er.CONCAT_STATES, er.ACTION_REPEAT] + RESIZE)
    frame_to_state = tf.transpose(frame_to_state, [0, 2, 3, 1])# Move er.ACTION_REPEAT into channels
else:
    frame_to_state = tf.reshape(ph.frame, [er.CONCAT_STATES] + STATE_DIM)

app = Struct(policy_index=0, quit=False, mb_count=0, print_action=FLAGS.inst==1, show_state_image=False,
    draw_attention=False, wireframe=True, pause=False)
if FLAGS.inst:
    FIRST_SEQ = (FLAGS.inst-1)*FLAGS.seq_per_inst
else:
    FIRST_SEQ = -FLAGS.seq_keep
    def init_vars(): sess.run(tf.global_variables_initializer())
    if FLAGS.record or FLAGS.replay or FLAGS.recreate_states:
        # Record new arrays
        app.policy_index = -1
    else:
        training.seq_recorded = FLAGS.seq_keep
training.append_batch = FIRST_SEQ

ops = Struct(
    per_minibatch=[], post_minibatch=[],
    per_update=[], post_update=[],
    new_batches=[],
    per_inst=[frame_to_state], post_inst=[])
def accum_value(value):
    accum = tf.Variable(tf.zeros_like(value), trainable=False)
    ops.per_minibatch.append(accum.assign_add(value))
    ops.post_update.append(accum.assign(tf.zeros_like(value)))
    return accum
def apply_gradients(grads, opt, accum_until_update=False, clip_norm=1.):
    grads, weights = zip(*grads)
    if accum_until_update:
        grads = [accum_value(g) for g in grads]
    # Clip gradients by global norm to prevent destabilizing policy
    grads,global_norm = tf.clip_by_global_norm(grads, clip_norm)
    grads = zip(grads, weights)
    (ops.per_update if accum_until_update else ops.per_minibatch).append(opt.apply_gradients(grads))
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
    if not BATCH_NORM: return x
    norm = tf.layers.BatchNormalization(scale=False, center=False, momentum=0.9)
    training_idx = 1
    x = [norm.apply(x[i], training=i==training_idx) for i in range(len(x))]
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
            [ops.post_inst, ops.post_minibatch][i] += [initial_state[c].assign(final_state[c]) for c in range(2)]
            if i==1: ops.new_batches += [initial_state[c].assign(tf.zeros_like(initial_state[c])) for c in range(2)]
    return x

CDF_80pc = 0.85
def make_fc(x, actions):
    with tf.variable_scope('hidden1'):
        x = layer_dense(x, FC_UNITS)
        x = layer_batch_norm(x)
        x = [heaviside(n) for n in x]
    with tf.variable_scope('hidden2'):
        if USE_LSTM: x = layer_lstm(x, FC_UNITS, ac)
        else: x = layer_dense(x, FC_UNITS)

        x = layer_batch_norm(x)
        if actions:
            with tf.variable_scope('actions'):
                actions = layer_dense(actions, FC_UNITS)
                x = [n+a for n,a in zip(x,actions)]

        # Offset controls cdf probability of activation being non-zero after batch norm.
        x = [heaviside(n-CDF_80pc) for n in x]
    return x

def make_conv_net(x):
    LAYERS = [
        (32, 6, 3, 0),
        (32, 4, 4, 0),
    ]
    x = [tf.expand_dims(n,-1) for n in x]
    for l,(filters, width, stride, conv3d) in enumerate(LAYERS):
        with tf.variable_scope('conv_%i' % l):
            pool_stride = None
            #if not stride%2: pool_stride = [1,2,2,1]; stride /= 2 # Smoothen filter responses

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
            x = layer_batch_norm(x)
            x = [tf.nn.relu(n) for n in x]
        print(x[0].shape)
    return x

if MHDPA_LAYERS: from tflayers.mhdpa import *
def make_attention_net(x, ac):
    for l in range(MHDPA_LAYERS):
        with tf.variable_scope('mhdpa_%i' % l):
            if l == 0:
                mhdpa = MHDPA()
                x = [concat_coord_xy(n) for n in x]
            for i,n in enumerate(x):
                x[i], attention = mhdpa.apply(n)
                if i==0 and l==(MHDPA_LAYERS-1):
                    ac.per_inst.attention = attention[0] # Display agent attention
                if i==1: ac.per_minibatch.__dict__['attention_minmax_%i'%l] = tf.stack([tf.reduce_min(attention), tf.reduce_max(attention)])
    return x

def make_policy_output(hidden, iteration):
    with tf.variable_scope('output_%i' % iteration):
        with tf.variable_scope('mean'):
            if POLICY_SOFTMAX:
                return layer_dense(hidden, ACTION_DIMS, tf.nn.softmax)
            policy = layer_dense(hidden, ACTION_DIMS, tf.sigmoid)

        if ACTION_TANH: # Multiply specific axes with tanh
            with tf.variable_scope('tanh'):
                tanh = layer_dense(hidden, len(ACTION_TANH), tf.tanh)
            for i in range(len(policy)):
                mult = [tf.ones_like(tanh[i][:,0])]*ACTION_DIMS
                for j,t in enumerate(ACTION_TANH):
                    mult[t] = tanh[i][:,j]
                policy[i] *= tf.stack(mult, -1)

        with tf.variable_scope('stddev'):
            stddev = layer_dense(hidden, policy[0].shape[1], tf.nn.softplus)
        return policy, stddev

STATE_ACTION_SAMPLES = 100
def make_policy(shared):
    hidden = make_fc(shared.layers, None)

    POLICY_PDFs = 5
    mean, stddev = zip(*[make_policy_output(hidden, i) for i in range(POLICY_PDFs)])
    mean_choices = list(map(tf.stack, zip(*mean)))
    stddev_choices = list(map(tf.stack, zip(*stddev)))
    if POLICY_PDFs > 1:
        with tf.variable_scope('choice'):
            choice_dist = layer_dense(hidden, POLICY_PDFs, tf.distributions.Categorical)
            choice = [tf.one_hot(c.sample() if i==0 else c.mode(), POLICY_PDFs) for i,c in enumerate(choice_dist)]
    else:
        choice = [tf.ones(shape=[n.shape[0], 1]) for n in hidden]
    choice_transpose = [tf.transpose([c], [2,1,0]) for c in choice]
    mean = [tf.reduce_sum(m*c, 0) for m,c in zip(mean_choices, choice_transpose)]
    stddev = [tf.reduce_sum(s*c, 0) for s,c in zip(stddev_choices, choice_transpose)]

    policy = Struct(mean=mean, stddev=stddev,
        mean_choices=mean_choices, stddev_choices=stddev_choices,
        choice=choice, weights_nonshared=scope_vars())
    policy.weights = shared.weights + policy.weights_nonshared
    if FLAGS.inst:
        return policy

    log_prob = tf.minimum(10., tf.stack([tf.reduce_sum(tf.distributions.Normal
        (policy.mean_choices[1][c], policy.stddev_choices[1][c]).log_prob(ph.actions), -1) for c in range(POLICY_PDFs)], -1))
    log_prob += tf.stack([choice_dist[1].log_prob(c) for c in range(POLICY_PDFs)], -1) if POLICY_PDFs>1 else 0

    log_prob = tf.reduce_max(log_prob, -1)# Policy with max probability of ph.actions
    policy.log_prob = tf.maximum(log_prob, -10.) # Prevent divide-by-0 in ratio

    policy.next = mean[-len(training.nsteps):]
    policy.mb, policy.mb_stddev = mean[1], stddev[1]
    if STATE_ACTION_SAMPLES > 1:
        pdf = tf.distributions.Normal(policy.mb, policy.mb_stddev)
        sampled = tf.transpose(pdf.sample(STATE_ACTION_SAMPLES), [1, 0, 2])
        sampled = tf.reshape(sampled, [FLAGS.minibatch * STATE_ACTION_SAMPLES, -1])
        policy.mb_sampled = tf.clip_by_value(sampled, *ACTION_CLIP)
    else:
        policy.mb_sampled = policy.mb

    return policy

def multi_actions_pre(state, actions, idx, batch_size=FLAGS.minibatch, include_policy=True):
    num_actions = len(DISCRETE_ACTIONS)
    if not num_actions: return
    a = tf.tile(tf.expand_dims(MULTI_ACTIONS, 0), [batch_size, 1, 1])
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

def tile_actions(actions):
    if not ACTION_DISCRETE:
        TILES = 10
        actions = [tf.concat([a*TILES + t for t in range(-TILES+1,TILES)], -1) for a in actions]
        actions = [heaviside(a) for a in actions]
    return [2*a - 1 for a in actions]

gamma_nsteps = FLAGS.gamma**tf.cast(ph.nsteps, DTYPE)
TDC_STEPS = len(training.nsteps)
def make_qvalue(shared, policy):
    state = shared.layers
    if policy:
        actions = [policy.mean[0]]
        if not FLAGS.inst:
            state = [state[0], state[1], tf.tile(state[1], [STATE_ACTION_SAMPLES, 1])] + state[2:]
            actions += [ph.actions, policy.mb_sampled] + policy.next
        actions = tile_actions(actions)
    else:
        actions = None

    combined = make_fc(state, actions)
    with tf.variable_scope('output'):
        q = [n[:,0] for n in layer_dense(combined, 1)]

    # Q for all actions in agent's current state
    q[0] = multi_actions_post(q[0], 1)
    ret = Struct(inst_policy=q[0])
    if FLAGS.inst: return ret

    [ret.q, ret.state] = q[1:3]
    ret.state = tf.reduce_mean(tf.reshape(ret.state, [FLAGS.minibatch, STATE_ACTION_SAMPLES]), 1)

    next_state_value = q[-len(training.nsteps):]
    ret.nstep = ph.rewards[:,:,r] + gamma_nsteps*tf.stack(next_state_value, -1)

    ret.weights = shared.weights + scope_vars()
    with tf.variable_scope('error_value'):
        error_predict = layer_dense(combined, 1)[1][:,0]
        error_weights = scope_vars()

    def update_qvalue(target_value, ac=None):
        ret.td_error = target_value - ret.q

        repl = gradient_override(error_predict, ret.td_error-error_predict)
        grads = opt.error.compute_gradients(repl, error_weights)
        gnorm_error = apply_gradients(grads, opt.error)

        repl = gradient_override(ret.q, ret.td_error)
        grad_s = opt.td.compute_gradients(repl, ret.weights)
        # TDC averaged over all n-steps
        for i in range(TDC_STEPS):
            repl = gradient_override(next_state_value[i], -gamma_nsteps[:,i]*error_predict / TDC_STEPS)
            grad_s2 = opt.td.compute_gradients(repl, ret.weights)
            for i in range(len(grad_s)):
                (g, w), g2 = grad_s[i], grad_s2[i][0]
                grad_s[i] = (g+g2, w)
        gnorm = apply_gradients(grad_s, opt.td)

        if ac:
            ac.per_minibatch.gnorm_error = gnorm_error
            ac.per_minibatch.gnorm_qvalue = gnorm

    ret.update = update_qvalue
    return ret

def make_shared(x):
    print(x[0].shape)
    if CONV_NET: x = make_conv_net(x)
    else: x = layer_batch_norm(x)
    if MHDPA_LAYERS: x = make_attention_net(x, ac)

    x = [tf.layers.flatten(n) for n in x]
    print(x[0].shape)
    return Struct(layers=x, weights=scope_vars())

opt = Struct(td=tf.train.AdamOptimizer(FLAGS.learning_rate),
    policy=tf.train.AdamOptimizer(FLAGS.learning_rate/10),
    error=tf.train.AdamOptimizer(1))

def copy_networks(scope_to, scope_from, interp=1.):
    copy_vars = scope_vars(scope_to, True), scope_vars(scope_from, True)
    assert(len(copy_vars[0]) == len(copy_vars[1]))
    for to,w in zip(*copy_vars):
        ops.post_update.append(to.assign(interp*w + (1-interp)*to))

allac = []
def make_acrl():
    ac = Struct(per_minibatch=Struct(), per_update=Struct(), per_inst=Struct())
    allac.append(ac)

    # Move concat states to last dimension
    state_inputs = [tf.expand_dims(frame_to_state,0)] + ([] if FLAGS.inst else ph.states)
    idx = range(len(state_inputs[0].shape))
    idx = [0] + idx[2:]
    idx.insert(-1, 1)
    CONCAT_STATE_DIM = STATE_DIM[:]
    CONCAT_STATE_DIM[-1] *= er.CONCAT_STATES
    state_inputs = [tf.reshape(tf.transpose(n,idx), [-1]+CONCAT_STATE_DIM) for n in state_inputs]

    def make_networks():
        with tf.variable_scope('shared'): shared = make_shared(state_inputs)
        with tf.variable_scope('policy'): policy = make_policy(shared)
        with tf.variable_scope('value'):
            if not SHARED_LAYERS: shared = make_shared(state_inputs)
            with tf.variable_scope('1'): value1 = make_qvalue(shared, policy)
        return value1, policy

    with tf.variable_scope('old'):
        value1, policy = make_networks()

    ac.per_inst.policy = policy.mean[0][0]
    ac.per_inst.policy_stddev = policy.stddev[0][0]
    ac.per_inst.policy_choice = policy.choice[0][0]
    ac.per_inst.policy_value = value1.inst_policy[0]
    if FLAGS.inst: return

    with tf.variable_scope('new'):
        new_value1, new_policy = make_networks()

    '''
    importance_ratio = tf.reduce_prod(policy_pdf.prob(ph.actions), -1)
    importance_ratio /= tf.reduce_sum(importance_ratio) + 1.
    ac.per_minibatch.importance_ratio = tf.reduce_max(importance_ratio)
    '''

    target_value = tf.reduce_max(value1.nstep, -1) # Maximize over n-step returns
    new_value1.update(target_value, ac)

    value_grad = tf.gradients(value1.state, policy.mb)[0]
    if 0:#value_grad is not None:
        value_grad = tf.where(ph.ddpg, value_grad, tf.zeros_like(value_grad))
        repl = gradient_override(policy.mb, value_grad)
        weights_deterministic = [w for w in policy.weights if not any([name in w.name for name in ['stddev', 'choice']])]
        grads = opt.policy.compute_gradients(repl, weights_deterministic)
        ac.per_update.gnorm_policy_ddpg = apply_gradients(grads, opt.policy)

    return_value = tf.reduce_max(value1.nstep, -1)
    adv = return_value - value1.state
    adv = tf.maximum(0., adv) # Only towards better actions
    adv = tf.where(ph.adv, adv, tf.zeros_like(adv))

    ratio = tf.exp(new_policy.log_prob - policy.log_prob)
    cliprange = 0.2
    cost = -adv * tf.minimum(ratio, tf.clip_by_value(ratio, 1-cliprange, 1+cliprange)) # PPO objective
    #cost = gradient_override(policy.log_prob, adv)

    grads_adv = opt.policy.compute_gradients(cost, new_policy.weights)
    ac.per_minibatch.gnorm_policy_adv = apply_gradients(grads_adv, opt.policy)

    copy_networks('old', 'new')

    ac.per_minibatch.reward_sum = tf.reduce_sum(ph.rewards[:,0,r])
    ac.per_minibatch.policy_max = tf.reduce_max(policy.mb, 0)
    ac.per_minibatch.policy_min = tf.reduce_min(policy.mb, 0)
    ac.per_minibatch.action_max = tf.reduce_max(ph.actions, 0)
    ac.per_minibatch.action_min = tf.reduce_min(ph.actions, 0)
    ac.per_minibatch.td_error = new_value1.td_error
    ac.per_minibatch.td_error_sum = tf.reduce_sum(tf.abs(new_value1.td_error))
    ac.per_minibatch.max_log_prob = tf.reduce_max(new_policy.log_prob)
    ac.per_minibatch.max_ratio = tf.reduce_max(ratio)

for r in range(er.ER_REWARDS):
    with tf.variable_scope('ac'): make_acrl()

state = Struct(frames=np.zeros([er.CONCAT_STATES, er.ACTION_REPEAT] + FRAME_DIM),
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
                (['attention'] if app.draw_attention else []) +
                (['pause'] if app.pause else []))
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
            app.policy_index = min(len(allac)-1, int(k - ord('1')))
        elif k==ord('a'):
            app.print_action ^= True
        elif k==ord('s'):
            FLAGS.sample_action = 0. if FLAGS.sample_action else 1.
        elif k==ord('i'):
            app.show_state_image = True
        elif k==ord('v'):
            FLAGS.adv ^= True
        elif k==ord('g'):
            FLAGS.ddpg ^= True
        elif k==ord('t'):
            training.enable ^= True
        elif k==ord('k'):
            # Bootstrap learning with user-supplied trajectories, then turn them off
            FLAGS.seq_keep = 0
        elif k==ord('r'):
            state.next_reset = True
        elif k==ord('m'):
            app.draw_attention ^= True
        elif k==ord('w'):
            app.wireframe ^= True
        elif k==ord('p'):
            app.pause ^= True
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
    def choose_action(value): # Choose from Q-values or softmax policy
        return np.random.choice(value.shape[0], p=softmax(value)) if FLAGS.sample_action else np.argmax(value)
    def interp(f, a, b): return a + f*(b-a)

    a = action.keyboard[:ACTION_DIMS].copy()
    if ACTION_DISCRETE: a = onehot_vector(int(a[0]+1.), ACTION_DIMS)

    if state.count > 0 and app.policy_index != -1:
        stddev = app.per_inst.policy_stddev * FLAGS.sample_action
        a = np.random.normal(app.per_inst.policy, stddev)
        a = np.clip(a, *ACTION_CLIP)
    '''
    if FLAGS.sample_action:
        np.random.seed(0)
        offset = np.array([FLAGS.sample_action*math.sin(2*math.pi*(r + state.count/20.)) for r in np.random.rand(ACTION_DIMS)])
        a = np.clip(a+offset, -1, 1.)
    '''

    env_action = a
    if ACTION_DISCRETE:
        env_action = np.argmax(a)
        a = onehot_vector(env_action, ACTION_DIMS)
    action_to_save = a

    obs = state.last_obs
    reward_sum = 0.
    state.frames[:-1] = state.frames[1:]
    for frame in range(er.ACTION_REPEAT):
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

        obs, reward, state.done, info = env.step(env_action)
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
    return [reward_sum], action_to_save

def append_to_batch():
    save_paths = er.seq_paths(training.append_batch)
    if not FLAGS.inst and FLAGS.recreate_states:
        if not state.count:
            training.saved_batch = ermem.mmap_seq(save_paths, 'r', states=False)
        batch = training.saved_batch
        state.frames[-1] = batch.rawframes[state.count]
        save_reward, save_action = batch.rewards[state.count], batch.actions[state.count]
    else:
        save_reward, save_action = step_to_frames()

    r, app.per_inst = ops_run('per_inst', {ph.frame: state.frames})
    if app.print_action:
        ops_print(app.per_inst)
    save_state = r[0]
    if app.print_action:
        print(dict(save_action=save_action,
            #save_reward=save_reward
            ))
        os.system('clear')

    if app.show_state_image:
        app.show_state_image = False
        proc = multiprocessing.Process(target=imshow,
            args=([save_state[0,:,:,CHANNELS*i:CHANNELS*(i+1)] for i in range(er.ACTION_REPEAT)],))
        proc.start()

    temp_paths = er.seq_paths(FLAGS.inst, 'temp')
    if not training.temp_batch:
        training.temp_batch = ermem.mmap_seq(temp_paths, 'w+')
    batch = training.temp_batch
    batch.rawframes[state.count] = state.frames[-1]
    batch.states[state.count] = save_state[-1]
    batch.rewards[state.count] = save_reward
    batch.actions[state.count] = save_action

    state.count += 1
    if state.count == er.TRAJECTORY_LENGTH:
        if FLAGS.inst or training.seq_recorded < FLAGS.seq_keep:
            print('Replacing batch #%i' % training.append_batch)
            for a in batch.arrays: del a
            training.temp_batch = None

            # Rename inst batch files into server's ER batches.
            for k in save_paths.keys():
                src = temp_paths[k]
                dst = save_paths[k]
                os.system('rm -f ' + dst)
                os.system('mv ' + src + ' ' + dst)

        training.seq_recorded += 1
        training.append_batch += 1
        if FLAGS.inst and training.seq_recorded == FLAGS.seq_per_inst:
            training.seq_recorded = 0
            training.append_batch = FIRST_SEQ
        state.count = 0

def ops_finish(sname):
    named_ops = ops.__dict__[sname]
    keys = sorted(allac[0].__dict__[sname].__dict__.keys())
    named_ops += [tf.concat([[allac[r].__dict__[sname].__dict__[s]]
        for r in range(len(allac))], 0) for s in keys]

def ops_run(sname, feed_dict):
    named_ops = ops.__dict__[sname]
    post_ops = ops.__dict__[sname.replace('per_', 'post_')]

    r = sess.run(named_ops, feed_dict)
    sess.run(post_ops)

    policy_index = 0 if app.policy_index==-1 else app.policy_index
    keys = sorted(allac[0].__dict__[sname].__dict__.keys())
    d = {s: r[-i-1][policy_index] for i,s in enumerate(reversed(keys))}
    return r, Struct(name=sname, **d)

def ops_print(output_struct):
    np.set_printoptions(suppress=True, precision=6, sign=' ')
    d = output_struct.__dict__
    d_str = {k: str(v) for k,v in d.items() if k != 'name' and len(str(v).split('\n')) < 10} # Skip large arrays
    pprint({d['name']: d_str})

if not FLAGS.inst:
    init_vars()
    ops.per_minibatch += tf.get_collection(tf.GraphKeys.UPDATE_OPS) # batch_norm
    if FLAGS.summary:
        train_writer = tf.summary.FileWriter(FLAGS.summary, sess.graph)
        merged = tf.summary.merge_all()
        ops.per_minibatch.insert(0, merged)

    ops_finish('per_minibatch')
    ops_finish('per_update')
ops_finish('per_inst')

def train_minibatch(mb):
    # Upload & train minibatch
    feed_dict = {
        ph.adv: FLAGS.adv,
        ph.ddpg: FLAGS.ddpg,
        ph.actions: mb.actions,
        ph.rewards: mb.rewards,
        ph.nsteps: mb.nsteps}
    for i in range(len(training.nsteps)+1):
        feed_dict[ph.states[i]] = mb.states[i]
    r, per_minibatch = ops_run('per_minibatch', feed_dict)
    ops_print(per_minibatch)
    # Prioritized experience replay according to TD error.
    mb.td_error[:] = per_minibatch.td_error

    app.mb_count += 1
    pprint(dict(
        policy_updates=dict(adv=FLAGS.adv, ddpg=FLAGS.ddpg),
        nsteps=mb.nsteps[0,:], # Just show the first one
        minibatch=app.mb_count,
        rates=dict(learning_rate=FLAGS.learning_rate, tau=FLAGS.tau, gamma=FLAGS.gamma),
        batches=dict(keep=FLAGS.seq_keep,inst=FLAGS.seq_inst,minibatch=FLAGS.minibatch)))
    os.system('clear') # Scroll up to see status

    if FLAGS.summary and not app.mb_count%100:
        summary = r[0]
        train_writer.add_summary(summary, app.mb_count)

def train_update_policy():
    MB_PER_UPDATE = 5
    if app.mb_count >= MB_PER_UPDATE:
        if not app.mb_count % MB_PER_UPDATE:
            _, train_update_policy.output = ops_run('per_update', {})
        ops_print(train_update_policy.output)

ermem = er.ERMemory(training.nsteps, STATE_DIM, ACTION_DIMS, FRAME_DIM)
def rl_loop():
    if app.quit: return False
    if FLAGS.inst or not training.enable or training.seq_recorded < FLAGS.seq_keep:
        append_to_batch()
    else:
        if app.pause:
            time.sleep(0.1)
        else:
            mb = ermem.fill_mb()
            if mb != None:
                #if mb.step==0: sess.run(ops.new_batches) # Minibatch now has different steps
                train_minibatch(mb)
                train_update_policy()
        env_render() # Render needed for keyboard events
    return True
import utils; utils.loop_while(rl_loop)
