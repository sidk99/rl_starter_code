from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

# Transition = namedtuple('Transition', ('state', 'action', 'logprob', 'mask', 'next_state', 'reward', 'value'))
# SimpleTransition = namedtuple('SimpleTransition', ('state', 'action', 'logprob', 'mask','reward', 'value'))
# SimplerTransition = namedtuple('SimplerTransition', ('state', 'action', 'mask', 'reward'))
# InputOutput = namedtuple('InputOutput', ('loss'))
# RNTransition = namedtuple('StepTransition', ('state', 'action', 'logprob', 'mask', 'reward', 'value', 'step', 'task'))
# BanditTransition = namedtuple('BanditTransition', ('state', 'action', 'logprob', 'reward'))
# SimpleBanditTransition = namedtuple('SimpleBanditTransition', ('state', 'action', 'reward'))

# # probably what we need
# SimpleMaskedBanditTransition = namedtuple('SimpleMaskedBanditTransition', ('state', 'action', 'mask', 'reward'))
# SimplerQTransition = namedtuple('SimplerQTransition', ('state', 'action', 'mask', 'next_state', 'reward'))


class Memory(object):
    def __init__(self, element):
        self.memory = []
        if element == 'on_policy':
            self.element = namedtuple('OnPolicy', ('state', 'action', 'mask', 'reward'))
        elif element == 'off_policy':
            self.element = namedtuple('OnPolicy', ('state', 'action', 'mask', 'next_state', 'reward'))
        else:
            assert False

        # if element == 'transition':
        #     self.element = Transition
        # elif element == 'simpletransition':
        #     self.element = SimpleTransition
        # elif element == 'simplertransition':
        #     self.element = SimplerTransition
        # elif element == 'simplerqtransition':
        #     self.element = SimplerQTransition
        # elif element == 'inputoutput':
        #     self.element = InputOutput
        # elif element == 'rntransition':
        #     self.element = RNTransition
        # elif element == 'bandittransition':
        #     self.element = BanditTransition
        # elif element == 'simplebandittransition':
        #     self.element = SimpleBanditTransition
        # elif element == 'simplemaskedbandittransition':
        #     self.element = SimpleMaskedBanditTransition
        # else:
        #     assert False

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