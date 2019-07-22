import torch.nn as nn
import torch.nn.functional as F

class SimpleValueFn(nn.Module):
    def __init__(self, state_dim, hid_dim):
        super(SimpleValueFn, self).__init__()
        self.value_affine = nn.Linear(state_dim, hid_dim)
        self.value_head = nn.Linear(hid_dim, 1)

    def forward(self, state):
        state_values = self.value_head(F.relu(self.value_affine(state)))
        return state_values


class ValueFn(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value