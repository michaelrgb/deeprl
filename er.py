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

def seq_paths(seq_num, path=None):
    if not path:
        path = '%s_%i' % (ENV_NAME, ACTION_REPEAT)
    os.system('mkdir -p sequences; mkdir -p sequences_keep')
    path = ('sequences_keep/' if seq_num<0 else 'sequences/') + path + '_%i_%s.mmap'
    seq_num = -seq_num - 1 if seq_num<0 else seq_num
    return {key: path % (seq_num, key) for key in ['rawframes', 'states', 'actions', 'rewards']}

class ERMemory:
    def __init__(self, nsteps, state_dim, action_dims, frame_dim):
        self.nsteps = nsteps
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.frame_dim = frame_dim

        self.proclist = []
        self.current_step = 0
        self.mb = None
        self.mb_replace_size = FLAGS.minibatch / 4 # Amount of minibatch to replace each iteration

        manager = multiprocessing.Manager()
        self.seq_sets = manager.list()

    def mmap_seq(self, paths, mode, only_actions=False, states=True, rawframes=True):
        seq = Struct(actions=np.memmap(paths['actions'], DTYPE.name, mode, shape=(TRAJECTORY_LENGTH, self.action_dims)))
        if only_actions:
            return seq.actions
        seq.rewards = np.memmap(paths['rewards'], DTYPE.name, mode, shape=(TRAJECTORY_LENGTH, ER_REWARDS))
        seq.arrays = [seq.actions, seq.rewards]
        if states:
            seq.states = np.memmap(paths['states'], DTYPE.name, mode, shape=(TRAJECTORY_LENGTH,) + tuple(self.state_dim))
            seq.arrays.append(seq.states)
        if rawframes:
            seq.rawframes = np.memmap(paths['rawframes'], DTYPE.name, mode, shape=(TRAJECTORY_LENGTH, ACTION_REPEAT) + tuple(self.frame_dim))
            seq.arrays.append(seq.rawframes)
        return seq

    def _proc_seq_set(self):
        seq_set = {}
        shuffled = list(range(-1,-(FLAGS.seq_keep+1),-1)) + list(range(FLAGS.seq_inst))
        np.random.shuffle(shuffled)
        while len(seq_set) < self.mb_replace_size:
            try:
                if not len(shuffled): return
                idx = shuffled.pop()
                paths = seq_paths(idx)
                seq = self.mmap_seq(paths, 'r', rawframes=False)
                if any(np.isnan(seq.actions).flatten()):
                    print('NAN ACTIONS in ' + paths)
                    exit
                seq_set[idx] = seq
            except Exception as e:
                print(e)
                continue
        self.seq_sets.append(seq_set.values())
    def _get_new_seq_set(self):
        if 1:
            PROCESSES = 2
            while len(self.proclist) < PROCESSES:
                proc = multiprocessing.Process(target=self._proc_seq_set)
                proc.start()
                self.proclist.append(proc)
            proc = next((proc for proc in self.proclist if not proc.is_alive()), None)
            if proc:
                proc.join()
                self.proclist.remove(proc)
        else:
            self._proc_seq_set()
        self.seq_set = self.seq_sets.pop() if len(self.seq_sets) else None

    def _fill_sub_mb(self, seq_step, mb_indices):
        assert(len(mb_indices) == self.mb_replace_size)

        seq_step += CONCAT_STATES
        mb = self.mb
        for b,mb_idx in enumerate(mb_indices):# indices of entries to replace
            seq = self.seq_set[b]

            mb.priority[mb_idx] = -float('inf')
            mb.states[0][mb_idx] = seq.states[seq_step-CONCAT_STATES:seq_step]
            mb.actions[mb_idx] = seq.actions[seq_step]
            for i,nsteps in enumerate(self.nsteps):
                accum_reward = 0.
                for n in range(nsteps):
                    next_step = seq_step + n
                    accum_reward += seq.rewards[next_step] * FLAGS.gamma**n
                    next_step += 1
                    if next_step == TRAJECTORY_LENGTH:
                        break
                mb.states[i+1][mb_idx] = seq.states[next_step-CONCAT_STATES:next_step]
                mb.rewards[mb_idx, i] = accum_reward
                mb.nsteps[mb_idx, i] = next_step - seq_step

    def fill_mb(self): # Each minibatch is random subset of batch trajectories
        seq_step = self.current_step
        if seq_step+CONCAT_STATES == TRAJECTORY_LENGTH-1:
            seq_step = 0

        if seq_step == 0:
            self._get_new_seq_set()
            if not self.seq_set: return None
        self.current_step = seq_step+1

        first_mb = False
        if not self.mb:
            first_mb = True
            self.mb = Struct(
                priority = np.zeros([FLAGS.minibatch]),
                actions = np.zeros([FLAGS.minibatch, self.action_dims]),
                states = [np.zeros([FLAGS.minibatch, CONCAT_STATES] + self.state_dim) for n in self.nsteps+[1]],
                rewards = np.zeros([FLAGS.minibatch, len(self.nsteps), ER_REWARDS]),
                nsteps = np.zeros([FLAGS.minibatch, len(self.nsteps)]))

        if first_mb:
            # Fill the entire minibatch
            for i in range(FLAGS.minibatch / self.mb_replace_size):
                self._fill_sub_mb(seq_step, range(i*self.mb_replace_size, (i+1)*self.mb_replace_size))
        else:
            # Sample minibatch indices to replace
            all_priorities = {k: self.mb.priority[k] for k in range(FLAGS.minibatch)}
            mb_indices = []
            for i in range(self.mb_replace_size):
                idx, priority = zip(*all_priorities.items())
                scale = -1.
                probs = softmax(np.array([scale*p for p in priority]))
                chosen = idx[np.random.choice(len(idx), p=probs)]
                all_priorities.pop(chosen)
                mb_indices.append(chosen)
            self._fill_sub_mb(seq_step, mb_indices)
        return self.mb
