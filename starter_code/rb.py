from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

OnPolicy = namedtuple('OnPolicy', ('state', 'action', 'mask', 'reward'))
OffPolicy = namedtuple('OffPolicy', ('state', 'action', 'mask', 'next_state', 'reward'))


class Memory(object):
    def __init__(self, element):
        self.memory = []
        if element == 'on_policy':
            self.element = OnPolicy
        elif element == 'off_policy':
            self.element = OffPolicy
        else:
            assert False

    def push(self, **kwargs):
        """Saves a transition.""" 
        self.memory.append(self.element(**kwargs))

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.element(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return self.element(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

    def clear_buffer(self):
        self.memory = []


"""
    The big difference between OffPolicyMemory and OnPolicyMemory is that
    OnPolicyMemory clears out the data after it has been used
"""

class OffPolicyMemory(Memory):
    def __init__(self, max_replay_buffer_size):
        super(OffPolicyMemory, self).__init__('off_policy')
        self.max_replay_buffer_size = max_replay_buffer_size

    # probably this could be a LOT faster
    def push(self, *args):
        if len(self.memory) > max_replay_buffer_size:
            self.memory = self.memory[1:]  # pop front
        super(OffPolicyMemory, self).push(*args)


class OnPolicyMemory(Memory):
    def __init__(self):
        super(OnPolicyMemory, self).__init__('on_policy')




class StaticMemory():
    def __init__(self, max_replay_buffer_size, ob_dim, action_dim):
        self._states = np.empty((max_replay_buffer_size, ob_dim))
        self._actions = np.empty((max_replay_buffer_size, action_dim))
        self._masks = np.empty((max_replay_buffer_size, 1), dtype='uint8')
        self._rewards = np.empty((max_replay_buffer_size, 1))

        self._top = 0
        self._size = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def push(self, state, action, mask, reward):
        self._states[self._top] = state
        self._actions[self._top] = action
        self._masks[self._top] = mask
        self._rewards[self._top] = reward
        self._advance()

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = max_replay_buffer_size
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            state=self._states[indices],
            action=self._actions[indices],
            mask=self._masks[indices],
            reward=self._rewards[indices],
            )
        return batch

    def __len__(self):
        return self._size

    def clear_buffer(self):
        pass




