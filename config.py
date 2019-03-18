import tensorflow as tf
import os
import er
from utils import *

flags = tf.app.flags
flags.DEFINE_integer('inst', 0, 'ID of agent that accumulates gradients on server')
flags.DEFINE_integer('seq_keep', 0, 'Sequences recorded from user actions')
flags.DEFINE_integer('seq_inst', 128, 'Sequences in queue recorded from agents')
flags.DEFINE_integer('seq_per_inst', 64, 'Sequences recorded per agent instance')
flags.DEFINE_integer('minibatch', 64, 'Minibatch size')
flags.DEFINE_integer('update_mb', 10, 'Minibatches per policy update')
flags.DEFINE_boolean('replay', False, 'Replay actions recorded in memmap array')
flags.DEFINE_boolean('recreate_states', False, 'Recreate kept states from saved raw frames')
flags.DEFINE_boolean('record', False, 'Record over kept sequences')
flags.DEFINE_string('summary', '/tmp/tf', 'Summaries path for Tensorboard')
flags.DEFINE_string('env_seed', '', 'Seed number for new environment')
flags.DEFINE_float('sample_action', 0., 'Sample actions, or use modal policy')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_float('gamma', 0.99, 'Discount rate')
flags.DEFINE_integer('nsteps', 30, 'Multi-step returns for Q-function')
FLAGS = flags.FLAGS

PORT, PROTOCOL = 'localhost:2222', 'grpc'
if not FLAGS.inst:
    server = tf.train.Server({'local': [PORT]}, protocol=PROTOCOL, start=True)
sess = tf.InteractiveSession(PROTOCOL+'://'+PORT)

LSTM_UNROLL = 0
MHDPA_LAYERS = 0#3
FC_UNITS = 256

FIXED_ACTIONS = []
ENV_NAME = os.getenv('ENV')
er.ENV_NAME = ENV_NAME
if not ENV_NAME:
    raise Exception('Missing ENV environment variable')
if ENV_NAME == 'CarRacing-v0':
    import gym.envs.box2d
    car_racing = gym.envs.box2d.car_racing
    car_racing.WINDOW_W = 800 # Default is huge
    car_racing.WINDOW_H = 600
    FIXED_ACTIONS = [
        [-1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 0.]]
elif ENV_NAME == 'MountainCarContinuous-v0':
    FIXED_ACTIONS = [[-1.], [1.]]
elif ENV_NAME == 'FlappyBird-v0':
    import gym_ple # [512, 288]
elif 'Bullet' in ENV_NAME:
    import pybullet_envs

import gym
env = gym.make(ENV_NAME)
env._max_episode_steps = None # Disable step limit
envu = env.unwrapped

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

training = Struct(enable=True, seq_recorded=0, batches_mtime={}, temp_batch=None)
ph = Struct(
    actions=tf.placeholder(DTYPE, [FLAGS.minibatch, ACTION_DIMS]),
    states=tf.placeholder(DTYPE, [FLAGS.minibatch, er.CONCAT_STATES] + STATE_DIM),
    rewards=tf.placeholder(DTYPE, [FLAGS.minibatch, er.ER_REWARDS]),
    frame=tf.placeholder(DTYPE, [er.CONCAT_STATES, er.ACTION_REPEAT] + FRAME_DIM),
    mb_count=tf.placeholder('int32', []),
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
