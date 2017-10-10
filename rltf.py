import tensorflow as tf, numpy as np
from utils import *
sess = tf.InteractiveSession()

MAX_EPISODE_TRANSITIONS = 50
STATE_FRAMES = 4
FRAME_DIM = [96, 96, 3]
STATE_DIM = FRAME_DIM[:2] + [FRAME_DIM[-1]*STATE_FRAMES]

def make_layer(x, conv_width, input_channels, output_channels):
    with tf.name_scope('weights'):
        W_conv = weight_variable([conv_width, conv_width, input_channels, output_channels])
        variable_summaries(W_conv)
    with tf.name_scope('bias'):
        b_conv = bias_variable([output_channels])
        variable_summaries(b_conv)
    x = [lrelu(conv2d(i, W_conv) + b_conv) for i in x]
    x = [max_pool(i, size=5, stride=2) for i in x]
    return x

def make_net(x):
    with tf.name_scope('layer0'):
        x = make_layer(x, 5, STATE_DIM[-1], 32)
    with tf.name_scope('layer1'):
        x = make_layer(x, 5, 32, 64)
    with tf.name_scope('layer2'):
        x = make_layer(x, 5, 64, 128)
    return x

def make_layer_flat(x, input_size, input_channels, output_channels):
    flat_size = input_size * input_size * input_channels
    with tf.name_scope('layer_flat'):
        with tf.name_scope('weights'):
            W_f = weight_variable([flat_size, output_channels])
            variable_summaries(W_f)
        with tf.name_scope('bias'):
            b_f = bias_variable([output_channels])
            variable_summaries(b_f)
        x = [tf.reshape(i, [-1, flat_size]) for i in x]
        x = [lrelu(tf.matmul(i, W_f) + b_f) for i in x]
    return x

tf.set_random_seed(0)

ER_SIZE = 50
state_tf = tf.placeholder(DTYPE, shape=[None] + STATE_DIM)
er_state = tf.get_variable("er_state", [ER_SIZE] + STATE_DIM,           dtype=DTYPE, trainable=False)
er_state_next = tf.get_variable("er_state_next", [ER_SIZE] + STATE_DIM, dtype=DTYPE, trainable=False)
er_reward = tf.get_variable("er_reward", [ER_SIZE, 1],                  dtype=DTYPE, trainable=False)

net = make_net([er_state, er_state_next])

# Calculate topmost convolution dimensionality
sess.run(tf.global_variables_initializer())
eval = net[0].eval(feed_dict={})
[q_tf, q_next_tf] = make_layer_flat(net, input_size=eval.shape[1], input_channels=eval.shape[3], output_channels=1)

with tf.name_scope('loss'):
    # SARSA
    GAMMA = 0.9
    q_next_tf = tf.stop_gradient(q_next_tf)
    loss_tf = (er_reward + GAMMA*q_next_tf - q_tf)
    loss_tf = loss_tf * loss_tf
    #loss_tf = tf.pow(loss_tf, 2)
    loss_sum = tf.reduce_sum(loss_tf)

    loss_argmin = tf.argmin(loss_tf, 0)

    tf.summary.histogram('histogram', loss_tf)
    tf.summary.scalar('sum', loss_sum)

sess.run(tf.global_variables_initializer())
eval = q_tf.eval(feed_dict={})# TEST

opt = tf.train.GradientDescentOptimizer(1e-8)
#grads = opt.compute_gradients(loss_sum)
train_step = opt.minimize(loss_sum)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)

def gym_loop():
    import gym
    env = gym.make('CarRacing-v0')
    env.reset(); env.render()# Seem to need at least 1 reset&render before we get valid observation

    state = np.zeros([STATE_FRAMES] + FRAME_DIM, 'uint8')
    prev_state = np.empty_like(state)

    render_env = False
    show_obs = False
    episode = 0
    training_iter = 0
    while 1:
        # New episode
        observation = env.reset()

        for trans in range(MAX_EPISODE_TRANSITIONS):
            action = env.action_space.sample() # take a random action
            action = [0, 1, 0]

            reward_sum = 0.
            for frame in range(STATE_FRAMES):
                if show_obs:
                    print('Frame', trans); imshow(observation)
                if render_env:
                    env.render()# Not needed for state_pixels

                state[frame] = observation
                observation, reward, done, info = env.step(action)
                reward_sum += reward

            if trans > 0:
                save_index = -1
                if training_iter < ER_SIZE:
                    save_index = training_iter
                else:
                    summary, loss_sum_eval, loss_argmin_eval, _ = sess.run([merged, loss_sum, loss_argmin, train_step], feed_dict={})
                    save_index = loss_argmin_eval[0]
                    print(loss_sum_eval, save_index)
                    #loss_eval = sess.run(loss_tf); print(loss_eval)
                    train_writer.add_summary(summary, training_iter)

                if save_index != -1:# and training_iter < 100:
                    sess.run([er_state[save_index].assign(prev_state.reshape(STATE_DIM)/255.),
                             er_state_next[save_index].assign(state.reshape(STATE_DIM)/255.),
                             er_reward[save_index].assign([reward_sum])])

            prev_state[:] = state
            training_iter += 1

        episode += 1

gym_loop()
