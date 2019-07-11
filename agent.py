import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from rb import Memory

class Agent(nn.Module):
    def __init__(self, policy, valuefn, args):
        super(Agent, self).__init__()
        self.policy = policy
        self.valuefn = valuefn
        self.args = args

        self.initalize_memory()
        self.initialize_optimizer()

    def initalize_memory(self):
        self.buffer = Memory(element='simpletransition')

    def initialize_optimizer(self):
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.args.lr)
        self.value_optimizer = optim.Adam(self.valuefn.parameters(), lr=10*self.args.lr)

    def forward(self, state, deterministic):
        state = torch.from_numpy(state).float()
        action = self.policy.select_action(state, deterministic)
        log_prob = self.policy.get_log_prob(state, action)
        value = self.valuefn(state)
        if action.dim() == 0: 
            action = action.item()
        else:
            action = action.detach().numpy()
        return action, log_prob, value

    def store_transition(self, transition):
        self.buffer.push(
            transition['state'],
            transition['action'],
            transition['logprob'],
            transition['mask'],
            transition['reward'],
            transition['value'])

