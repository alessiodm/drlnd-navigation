import numpy as np
from collections import deque, namedtuple
import torch

# Hyperparameters for Prioritized Experience Replay
E = 0.01
A = 0.6
B = 0.4
B_INC_PER_SAMPLE = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, PER=False):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            PER (bool): whether to use Prioritized Experience Replay
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.memory = deque(maxlen=buffer_size)
        self.PER = PER
        self.ps = np.full((buffer_size,), E)
        self.b = B

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        cur_len = len(self.memory)
        self.memory.append(e)
        if cur_len == self.buffer_size:
            np.roll(self.ps, 1)
        self.ps[cur_len - 1] = E

    def sample(self):
        """Sample a batch of experiences from memory."""
        if self.PER:
            # This sampling is O(N) and not very efficient when the memory is large.
            # We can improve it to O(logN) with a sum-tree data-structures, see:
            #   https://stackoverflow.com/a/58780640
            # But for this implementation, we use Numpy as much as possible and we
            # resemble the lecture slides and formulas.
            all_indices, probs = self.__get_Ps()
            selection = np.random.choice(all_indices, p=probs, size=self.batch_size)
            Ps = probs[selection]
            importance_sampling_multiplier = ( (1. / len(selection)) * (1. / Ps) ) ** self.b
            self.b = np.min([1., self.b + B_INC_PER_SAMPLE]) # grow `b` linearly.
        else:
            all_indices = np.arange(len(self.memory))
            selection = np.random.choice(all_indices, size=self.batch_size)
            importance_sampling_multiplier = 1.0
        values = self.__unpack(selection)
        return selection, values, importance_sampling_multiplier

    def update_p(self, selection, deltas):
        """Update the buffered experiences priorities according to the absolute deltas.

        Params:
        =======
            selection: Numpy array of indices for the experiences to update
            deltas: Numpy array of delta to update for each experience
        """
        assert self.PER, "Update should be called only on Prioritized Experience Replay mode"
        self.ps[selection] = np.abs(deltas.reshape(len(deltas),)) + E

    def __get_Ps(self):
        """Compute the Pi for all the experiences in the current buffer."""
        # We use `indices` b/c at the beginning the memory is partially filled.
        all_indices = np.arange(len(self.memory))
        return all_indices, (self.ps[all_indices] ** A) / np.sum(self.ps[all_indices] ** A)

    def __unpack(self, selection):
        """Unpack a collection of namedtuple experiences retrieved from the memory buffer.
        
        Params:
        =======
            selection: the indices for which retriving the experiences from memory.
        """ 
        experiences = [self.memory[i] for i in selection]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
