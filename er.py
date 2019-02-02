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

        self.buffer_size = max(FLAGS.minibatch, 200)
        # Should not replace more of buffer than we have evalulated via a minibatch
        self.buffer_replace_size = FLAGS.minibatch

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
        while len(seq_set) < self.buffer_replace_size:
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
    def _new_seq_set(self):
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

    def _fill_buffer(self, seq_step, buf_indices): # buf_indices to replace
        seq_step += CONCAT_STATES
        buf = self.buffer
        for b,buf_idx in enumerate(buf_indices):
            seq = self.seq_set[b % len(self.seq_set)] # % only for filling the entire initial buffer

            buf.states[0][buf_idx] = seq.states[seq_step-CONCAT_STATES:seq_step]
            buf.actions[buf_idx] = seq.actions[seq_step]
            for i,nsteps in enumerate(self.nsteps):
                accum_reward = 0.
                for n in range(nsteps):
                    next_step = seq_step + n
                    accum_reward += seq.rewards[next_step] * FLAGS.gamma**n
                    next_step += 1
                    if next_step == TRAJECTORY_LENGTH:
                        break
                buf.states[i+1][buf_idx] = seq.states[next_step-CONCAT_STATES:next_step]
                buf.rewards[buf_idx, i] = accum_reward
                buf.nsteps[buf_idx, i] = next_step - seq_step

    def _alloc_batch(self, size):
        return Struct(
            priority = np.zeros([size]),
            buffer_idx = np.zeros([size], np.int),
            actions = np.zeros([size, self.action_dims]),
            states = [np.zeros([size, CONCAT_STATES] + self.state_dim) for n in self.nsteps+[1]],
            rewards = np.zeros([size, len(self.nsteps), ER_REWARDS]),
            nsteps = np.zeros([size, len(self.nsteps)]))

    # Fill entire minibatch from buffer, which is replaced by sampling for lowest buffer priorities.
    def fill_mb(self):
        seq_step = self.current_step
        if seq_step+CONCAT_STATES == TRAJECTORY_LENGTH-1:
            seq_step = 0

        if seq_step == 0:
            self._new_seq_set()
            if not self.seq_set: return None
        self.current_step = seq_step+1

        if not self.mb:
            self.mb = self._alloc_batch(FLAGS.minibatch)
            self.buffer = self._alloc_batch(self.buffer_size)

            # Fill the entire buffer
            self._fill_buffer(seq_step, range(self.buffer_size))
        else:
            # Copy previous minibatch priorities into buffer
            for mb_idx in range(FLAGS.minibatch):
                self.buffer.priority[self.mb.buffer_idx[mb_idx]] = self.mb.priority[mb_idx]

            # Sample lowest priority indices from buffer to replace
            all_priorities = {k: -self.buffer.priority[k] for k in range(self.buffer_size)}
            buf_indices = []
            for i in range(self.buffer_replace_size):
                idx, priority = zip(*all_priorities.items())
                probs = softmax(np.array(priority))
                chosen = idx[np.random.choice(len(idx), p=probs)]
                all_priorities.pop(chosen)
                buf_indices.append(chosen)
            #print(self.mb.buffer_idx, buf_indices)
            self._fill_buffer(seq_step, buf_indices)

        mb, buf = self.mb, self.buffer
        buf_idx = self.mb.buffer_idx[-1]
        for mb_idx in range(FLAGS.minibatch):
            # Replace entire minibatch by looping over buffer
            buf_idx = (buf_idx+1) % self.buffer_size

            mb.buffer_idx[mb_idx] = buf_idx
            mb.actions[mb_idx] = buf.actions[buf_idx]
            mb.rewards[mb_idx] = buf.rewards[buf_idx]
            mb.nsteps[mb_idx] = buf.nsteps[buf_idx]
            for n in range(len(self.nsteps)+1):
                mb.states[n][mb_idx] = buf.states[n][buf_idx]

        return self.mb
