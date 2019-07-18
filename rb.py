from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb
Transition = namedtuple('Transition', ('state', 'action', 'logprob', 'mask', 'next_state', 'reward', 'value'))
SimpleTransition = namedtuple('SimpleTransition', ('state', 'action', 'logprob', 'mask','reward', 'value'))
SimplerTransition = namedtuple('SimplerTransition', ('state', 'action', 'mask','reward'))
InputOutput = namedtuple('InputOutput', ('loss'))
RNTransition = namedtuple('StepTransition', ('state', 'action', 'logprob', 'mask', 'reward', 'value', 'step', 'task'))
BanditTransition = namedtuple('BanditTransition', ('state', 'action', 'logprob', 'reward'))
SimpleBanditTransition = namedtuple('SimpleBanditTransition', ('state', 'action', 'reward'))
SimpleMaskedBanditTransition = namedtuple('SimpleMaskedBanditTransition', ('state', 'action', 'mask', 'reward'))

class Memory(object):
    def __init__(self, element='transition'):
        self.memory = []
        if element == 'transition':
            self.element = Transition
        elif element == 'simpletransition':
            self.element = SimpleTransition
        elif element == 'simplertransition':
            self.element = SimplerTransition
        elif element == 'inputoutput':
            self.element = InputOutput
        elif element == 'rntransition':
            self.element = RNTransition
        elif element == 'bandittransition':
            self.element = BanditTransition
        elif element == 'simplebandittransition':
            self.element = SimpleBanditTransition
        elif element == 'simplemaskedbandittransition':
            self.element = SimpleMaskedBanditTransition
        else:
            assert False

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(self.element(*args))

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
        del self.memory[:]

    # # def unpack(self, batch):
    # def unpack_batch(self, batch, key, device):
    #     return torch.from_numpy(np.stack(batch._fields)).to(torch.float32).to(self.device)
    #     states = torch.from_numpy(np.stack(batch.state)).to(torch.float32).to(self.device)  # (bsize, sdim)
    #     actions = torch.from_numpy(np.stack(batch.action)).to(torch.float32).to(self.device)  # (bsize, adim)
    #     masks = torch.from_numpy(np.stack(batch.mask)).to(torch.float32).to(self.device)  # (bsize)
    #     rewards = torch.from_numpy(np.stack(batch.reward)).to(torch.float32).to(self.device)  # (bsize)
    #     print('states')
    #     print(states.shape)
    #     print('actions')
    #     print(actions.shape)
    #     print('masks')
    #     print(masks.shape)
    #     print('rewards')
    #     print(rewards.shape)
    #     return states, actions, masks, rewards
