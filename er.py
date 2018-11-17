import tensorflow as tf
import sys, os, multiprocessing
from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS

TRAJECTORY_LENGTH = 100
# Frames an action is repeated for, combined into a state
ACTION_REPEAT = 3
# Concat frame states instead of repeating actions over multiple frames
CONCAT_STATES = 1

ER_REWARDS = 1

def batch_paths(batch_num, path=None):
    if not path:
        path = '%s_%i_%i' % (ENV_NAME, ACTION_REPEAT, 1)#ACTION_DOUBLE)
    os.system('mkdir -p batches')
    path = ('batches_keep/' if batch_num<0 else 'batches/') + path + '_%i_%s.mmap'
    return {key: path % (batch_num, key) for key in ['rawframes', 'states', 'actions', 'rewards']}

manager = multiprocessing.Manager()
batch_sets = manager.list()

class ERMemory:
    def __init__(self, mb_size, nsteps, state_dim, action_dims, frame_dim):
        self.mb_size = mb_size
        self.nsteps = nsteps
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.frame_dim = frame_dim

        self.proclist = []
        self.current_step = 0
        self.mb = None

    def mmap_batch(self, paths, mode, only_actions=False, states=True, rawframes=True):
        batch = Struct(actions=np.memmap(paths['actions'], DTYPE.name, mode, shape=(TRAJECTORY_LENGTH, self.action_dims)))
        if only_actions:
            return batch.actions
        batch.rewards = np.memmap(paths['rewards'], DTYPE.name, mode, shape=(TRAJECTORY_LENGTH, ER_REWARDS))
        batch.arrays = [batch.actions, batch.rewards]
        if states:
            batch.states = np.memmap(paths['states'], DTYPE.name, mode, shape=(TRAJECTORY_LENGTH,) + tuple(self.state_dim))
            batch.arrays.append(batch.states)
        if rawframes:
            batch.rawframes = np.memmap(paths['rawframes'], DTYPE.name, mode, shape=(TRAJECTORY_LENGTH, ACTION_REPEAT) + tuple(self.frame_dim))
            batch.arrays.append(batch.rawframes)
        return batch

    def proc_batch_set(self):
        batch_set = {}
        shuffled = list(range(-1,-(FLAGS.batch_keep+1),-1)) + list(range(FLAGS.batch_inst))
        np.random.shuffle(shuffled)
        while len(batch_set) < FLAGS.minibatch:
            try:
                if not len(shuffled): return
                idx = shuffled.pop()
                b = self.mmap_batch(batch_paths(idx), 'r', rawframes=False)
                batch_set[idx] = b
            except Exception as e:
                print(e)
                continue
        batch_sets.append(batch_set.values())

    def fill_mb(self): # Each minibatch is random subset of batch trajectories
        step = self.current_step
        if step+CONCAT_STATES == TRAJECTORY_LENGTH-1:
            step = 0

        if step == 0:
            if 1:
                PROCESSES = 2
                while len(self.proclist) < PROCESSES:
                    proc = multiprocessing.Process(target=self.proc_batch_set)
                    proc.start()
                    self.proclist.append(proc)
                proc = next((proc for proc in self.proclist if not proc.is_alive()), None)
                if proc:
                    proc.join()
                    self.proclist.remove(proc)
            else:
                self.proc_batch_set()
            if not len(batch_sets): return None
            self.batch_set = batch_sets.pop()
        self.current_step = step+1

        if not self.mb:
            self.mb = Struct(
                actions = np.zeros([self.mb_size, self.action_dims]),
                states = [np.zeros([self.mb_size, CONCAT_STATES] + self.state_dim) for n in self.nsteps+[1]],
                rewards = np.zeros([self.mb_size, len(self.nsteps), ER_REWARDS]),
                nsteps = list(range(len(self.nsteps))))
        mb = self.mb
        mb.step = step
        step += CONCAT_STATES
        for b,batch in enumerate(self.batch_set):
            mb.states[0][b] = batch.states[step-CONCAT_STATES:step]
            mb.actions[b] = batch.actions[step]
            for i,nsteps in enumerate(self.nsteps):
                accum_reward = 0.
                for n in range(nsteps):
                    next_step = step + n
                    accum_reward += batch.rewards[next_step] * FLAGS.gamma**n
                    next_step += 1
                    if next_step == TRAJECTORY_LENGTH:
                        break
                mb.rewards[b,i] = accum_reward
                mb.states[i+1][b] = batch.states[next_step-CONCAT_STATES:next_step]
                if b==0:
                    mb.nsteps[i] = next_step - step
        return mb
