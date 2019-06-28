import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.action_encoder = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        action_encoded = F.relu(self.action_encoder(x))
        action_scores = self.action_head(action_encoded)
        return F.softmax(action_scores, dim=-1)#, state_values


class ValueFn(nn.Module):
    def __init__(self):
        super(ValueFn, self).__init__()
        self.value_affine = nn.Linear(4, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        state_values = self.value_head(F.relu(self.value_affine(state)))
        return state_values