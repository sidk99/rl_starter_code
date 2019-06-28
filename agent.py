import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from rb import Memory

class Agent(nn.Module):
    def __init__(self, policy, valuefn):
        super(Agent, self).__init__()
        self.policy = policy
        self.valuefn = valuefn

        self.initalize_memory()
        self.initialize_optimizer()

    def initalize_memory(self):
        self.buffer = Memory(element='simpletransition')

    def initialize_optimizer(self):
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.valuefn.parameters(), lr=3e-4)

    def forward(self, x):
        probs, state_value = self.policy(x), self.valuefn(x)
        return probs, state_value

    def forward(self, state):
        state = torch.from_numpy(state).float()
        action_dist = self.policy(state)
        value = self.valuefn(state)
        m = Categorical(action_dist)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob, value

    def store_transition(self, transition):
        self.buffer.push(
            transition['state'],
            transition['action'],
            transition['logprob'],
            transition['mask'],
            transition['reward'],
            transition['value'])

