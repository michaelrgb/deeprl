GAMMA = 0.999
STATE_FRAMES = 4
ER_SIZE = 500
VALUE_LAYERS = 2
POLICY_LAYERS = 2
HIDDEN_NODES = 50
STATE_ACTIONS = False

import tensorflow as tf, numpy as np
from utils import *

LEARNING_RATE = 1e-3
opt_approx = tf.train.AdamOptimizer(LEARNING_RATE, epsilon=2)
opt_td = tf.train.AdamOptimizer(LEARNING_RATE/5, epsilon=2)
opt_policy = tf.train.AdamOptimizer(LEARNING_RATE/100, epsilon=2)

SAVE_SUMMARIES = False
SHOW_OBS = False
MAX_EPISODE_STATES = 10000

import gym
ENV_NAME = '''
CarRacing-v0
MountainCar-v0
'''.splitlines()[-1]
env = gym.make(ENV_NAME)
env._max_episode_steps = None # Disable step limit for now
envu = env.unwrapped
if ENV_NAME == 'CarRacing-v0':
    car_racing = gym.envs.box2d.car_racing
    car_racing.WINDOW_W = 800 # Default is huge
    car_racing.WINDOW_H = 600
elif ENV_NAME == 'MountainCar-v0':
    STATE_FRAMES = 1
ACTION_DIM = (env.action_space.shape or [env.action_space.n])[0]
FRAME_DIM = list(env.observation_space.shape)
if STATE_ACTIONS:
     FRAME_DIM[0] += ACTION_DIM

TEST_FEATURES = False
if TEST_FEATURES:
    # Ideal features for MountainCar
    FRAME_DIM[0] += 2

CONV_NET = len(FRAME_DIM) == 3
STATE_DIM = FRAME_DIM[:]
STATE_DIM[-1] *= STATE_FRAMES

sess = tf.InteractiveSession()

def test_lcn(image, batch_idx=0):
    frame_ph = tf.placeholder(tf.float32, [None] + list(image.shape[-3:]))
    lcn_op = local_contrast_norm(frame_ph, GAUSS_W)
    lcn = sess.run(lcn_op, feed_dict={frame_ph: np.expand_dims(image, 0)})
    return lcn[batch_idx]

REWARDS_GLOBAL = 1
with tf.name_scope('experience_replay'):
    np_state =      np.random.randn(*([ER_SIZE] + STATE_DIM)).astype(DTYPE.name)
    np_next_state = np.random.randn(*([ER_SIZE] + STATE_DIM)).astype(DTYPE.name)
    np_action =     np.random.randn(*([ER_SIZE, ACTION_DIM])).astype(DTYPE.name)
    np_reward =     np.random.randn(*([ER_SIZE, REWARDS_GLOBAL])).astype(DTYPE.name)
    er_state =      tf.Variable(np_state,       False, name='state')
    er_next_state = tf.Variable(np_next_state,  False, name='next_state')
    er_action =     tf.Variable(np_action,      False, name='action')
    er_reward =     tf.Variable(np_reward,      False, name='reward')

all_rewards = [er_reward[:, r] for r in range(REWARDS_GLOBAL)] + [
tf.abs(er_next_state[:, 1]), # Momentum
-tf.abs(er_next_state[:, 1]), # Stop
]

enable_training = False
def toggle_training():
    global enable_training
    enable_training ^= 1
    if not enable_training:
        return
    # Upload entire ER memory
    sess.run([er_state.assign(np_state),
              er_next_state.assign(np_next_state),
              er_action.assign(np_action),
              er_reward.assign(np_reward)])

def init_vars(): sess.run(tf.global_variables_initializer())
state_ph = tf.placeholder(tf.float32, [1] + STATE_DIM)

apply_ops = []

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
for r in range(len(all_rewards)):
    states, num_flat_inputs = [er_state, er_next_state, state_ph], STATE_DIM[0]
    shared_layer = states
    activation=lambda x: tf.nn.leaky_relu(x, alpha=1e-2)
    with tf.name_scope('conv_net'):
        value_weights = []

        conv_channels = STATE_DIM[-1]
        do_conv = CONV_NET
        if 0:#not do_conv:
            conv_channels = 1
            width_2d = 32
            shared_layer, value_weights = layer_fully_connected(shared_layer, num_flat_inputs, width_2d*width_2d, activation)
            shared_layer = [tf.reshape(i, [-1, width_2d, width_2d, 1]) for i in shared_layer]
            do_conv = True

        if do_conv:
            shared_layer, w = make_conv(shared_layer, conv_channels)
            value_weights += w
            # Calculate topmost convolution dimensionality to create fully-connected layer
            init_vars()
            shared_layer, num_flat_inputs = layer_reshape_flat(shared_layer, shared_layer[0].eval())

    def make_state_network(num_layers, num_outputs, last_layer_activation=None):
        weights = []
        prev_output = shared_layer
        input_size = num_flat_inputs
        for l in range(num_layers):
            prev_output, w = layer_fully_connected(prev_output, input_size, HIDDEN_NODES, activation)
            weights += w
            input_size = HIDDEN_NODES
        output, w = layer_fully_connected(prev_output, HIDDEN_NODES, num_outputs, last_layer_activation)
        weights += w
        return output, prev_output, weights

    with tf.name_scope('policy'):
        policy, _, policy_weights = make_state_network(POLICY_LAYERS, ACTION_DIM)
        all_policy_ph.append(policy[2])
        all_policy_minmax.append([tf.reduce_min(policy[0], axis=0), tf.reduce_max(policy[0], axis=0)])

    with tf.name_scope('state_value'):
        # Baseline value, independent of action a
        state_value, value_layer, w = make_state_network(VALUE_LAYERS, None)
        value_weights += w

    with tf.name_scope('local_advantage'):
        adv_layer, _, adv_weights = make_state_network(VALUE_LAYERS, HIDDEN_NODES*ACTION_DIM, activation)

        # Advantage is linear in (action-policy)
        off_policy = [er_action-i for i in policy]
        adv_layer = [tf.reshape(i, [-1, HIDDEN_NODES, ACTION_DIM]) * tf.expand_dims(j, 1) \
            for (i,j) in zip(adv_layer, off_policy)]
        adv_layer = [tf.reshape(i, [-1, HIDDEN_NODES*ACTION_DIM]) for i in adv_layer]

        adv_value, w = layer_fully_connected(adv_layer, HIDDEN_NODES*ACTION_DIM)
        adv_weights += w

        adv = adv_value[0]
        adv_grad = tf.gradients(adv, off_policy[0])[0]
        policy_grad = adv_grad * tf.expand_dims(adv, -1)

    with tf.name_scope('td_approx'):
        [td_approx], td_approx_weights = layer_fully_connected(value_layer[0], HIDDEN_NODES)
        td_approx_weights += value_weights

        [approx], w = layer_fully_connected(adv_layer[0], HIDDEN_NODES*ACTION_DIM)
        td_approx += approx
        td_approx_weights += adv_weights + w

    with tf.name_scope('q_value'):
        q_value = [i+j for i,j in zip(state_value, adv_value)]

    with tf.name_scope('td_error'):
        td_error = all_rewards[r] + GAMMA*q_value[1] - q_value[0]

        td_error_sq = td_error**2
        td_error_sum = tf.reduce_sum(td_error_sq)
        all_td_error_sum.append([td_error_sum])
        #loss_argmin = tf.argmin(td_error_sq, 0)
        loss_argmin = tf.constant([0])

    with tf.name_scope('td_approx'):
        td_approx_sum = tf.reduce_sum((td_error - td_approx)**2)
        tf.summary.scalar('loss', td_approx_sum)
        all_td_approx_sum.append([td_approx_sum])

    repl = gradient_override(policy[0], policy_grad)
    grad = opt_policy.compute_gradients(repl, var_list=policy_weights)
    apply_ops += [opt_policy.apply_gradients(grad)]

    # Approximate the TD error
    repl = gradient_override(td_approx, td_error-td_approx)
    grad = opt_approx.compute_gradients(repl, var_list=td_approx_weights)
    apply_ops += [opt_approx.apply_gradients(grad)]

    # TD error with gradient correction (TDC)
    for (value, weights) in [(q_value, value_weights+adv_weights)]:
        repl = gradient_override(value[0], td_error)
        grad_s = opt_td.compute_gradients(repl, var_list=weights)
        repl = gradient_override(value[1], -GAMMA*td_approx)
        grad_s2 = opt_td.compute_gradients(repl, var_list=weights)
        for i in range(len(grad_s)):
            g = grad_s[i][0] + grad_s2[i][0]
            grad_s[i] = (g, grad_s[i][1])
        apply_ops += [opt_td.apply_gradients(grad_s)]

td_error_sum = tf.concat(all_td_error_sum, axis=0)
td_approx_sum = tf.concat(all_td_approx_sum, axis=0)

if SAVE_SUMMARIES:
    merged = tf.summary.merge_all()
    apply_ops += [merged]
    train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)

init_vars()
policy_index = -1
def setup_key_actions():
    from pyglet.window import key
    a = np.array([0.]*3)
    def key_press(k, mod):
        global restart, policy_index
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
            toggle_training()
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

ACTIONS_CSV = 'actions_' + ENV_NAME + '_' + str(STATE_FRAMES) + '.csv'
try: actions_csv = np.loadtxt(open(ACTIONS_CSV, 'rb'), delimiter=',', skiprows=0)
except: actions_csv = []

def rl_loop():
    env.reset(); env.render()# Gym seems to need at least 1 reset&render before we get valid observation
    keyboard_action = setup_key_actions()

    er_num_saves = 0
    state = np.ones([STATE_FRAMES] + FRAME_DIM)
    next_state = np.ones_like(state)
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
            obs = list(obs)
            if STATE_ACTIONS:
                obs += list(action)
            if TEST_FEATURES:
                obs += [float(f > 0.) for f in [
                    obs[1],
                    -obs[1],
                    ]]
            #imshow([obs, test_lcn(obs)])
            if ENV_NAME == 'MountainCar-v0':
                # Mountain car env doesnt give any +reward
                if done:
                    reward = 1000.
            next_state[frame] = obs
            reward_sum += reward
        print('ER saves:', er_num_saves, 'State:', state, 'Action taken:', action, 'Reward:', reward_sum)
        return states_sofar + 1, reward_sum, done

    training_epochs = 0
    def run_training_ops():
        r = sess.run([loss_argmin, td_error_sum, td_approx_sum] + apply_ops)
        save_index=r[0][0]
        save_index = er_num_saves % ER_SIZE
        if SAVE_SUMMARIES:
            summary = r[-1]
            train_writer.add_summary(summary, training_epochs)
        r = dict(
            #save_index=save_index,
            #policy_weights='\n'.join(str(w.eval()) for w in policy_weights),
            training_epochs=training_epochs,
            td_error_sum=r[1],
            td_approx_sum=r[2],
            policy_index=policy_index,
            policy_action=policy_action,
            policy_minmax=sess.run(all_policy_minmax[policy_index]))
        for k in sorted(r.keys()):
            print(k + ': ' + str(r[k]))
        print(' ')
        return save_index

    global policy_index
    states_sofar = 0
    while 1:
        action = keyboard_action
        if not env.action_space.shape:
            # Convert a[0] values to one-hot vector
            action = [1. if action[0]+1. == i else 0. for i in range(ACTION_DIM)]

        policy_action = None
        if states_sofar > 0:
            # Calculate the policy actions:
            if policy_index != -1:
                [policy_action] = sess.run(all_policy_ph[policy_index], feed_dict={state_ph: [trans_state(state)]})
                action = policy_action

        do_env_step = True
        if enable_training:
            run_training_ops()
            training_epochs += 1
            # The GL env slows down training
            #if training_epochs % 10:
            do_env_step = False

        if do_env_step:
            if er_num_saves < len(actions_csv):
                action = actions_csv[er_num_saves]

            # action = [1., 0., 0] if next_state[0, 1] < 0. else [0., 0., 1.] # MountainCar: go with momentum
            states_sofar, reward_sum, done = step_state(states_sofar)
            if states_sofar >= 2:# and er_num_saves < ER_SIZE:
                save_index = er_num_saves % ER_SIZE
                if policy_index != -1:
                    # Keep some demonstrated actions
                    save_index = ER_SIZE/2 + er_num_saves % int(ER_SIZE/2)
                np_state[save_index] = trans_state(state)
                np_next_state[save_index] = trans_state(next_state)
                np_action[save_index] = action
                np_reward[save_index] = [reward_sum]

                er_num_saves += 1
                if er_num_saves == ER_SIZE:
                    toggle_training()

                    np.savetxt(ACTIONS_CSV, er_action[0:er_num_saves].eval(), delimiter=',')
                    # Switch to target policy
                    policy_index = 0

            state, next_state = next_state, state
            if done: states_sofar = 0
        else:
            # Render needed for keyboard events
            if not training_epochs % 5: env.render()

rl_loop()
