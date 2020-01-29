import torch
import torch.nn as nn
import torch.nn.functional as F

from starter_code.networks import MLP, CNN

class SimpleValueFn(nn.Module):
    def __init__(self, state_dim, hdim):
        super(SimpleValueFn, self).__init__()
        self.value_head = MLP(dims=[state_dim, *hdim, 1])

    def forward(self, state):
        state_values = self.value_head(state)
        return state_values

class SimpleQFn(nn.Module):
    def __init__(self, state_dim, action_dim, hdim):
        super(SimpleQFn, self).__init__()
        self.Q_head = MLP(dims=[state_dim+action_dim, *hdim, 1])

    def forward(self, state, action):
        raise NotImplementedError('need to concatenate action and state')
        Q_values = self.Q_head(state)
        return Q_values

class CNNValueFn(nn.Module):
    def __init__(self, state_dim):
        super(CNNValueFn, self).__init__()
        self.encoder = CNN(*state_dim)
        self.decoder = nn.Linear(self.encoder.image_embedding_size, 1)

    def forward(self, state):
        state_values = self.decoder(self.encoder(state))
        return state_values

class CNNQFn(nn.Module):
    def __init__(self, state_dim, action_dim, hdim=16):
        """
            hdim is for embedding action.
            So if the action is discrete hdim=16 should be large enough
        """
        super(CNNQFn, self).__init__()
        self.encoder = CNN(*state_dim)
        self.action_encoder = nn.Linear(action_dim, hdim)
        self.decoder = nn.Linear(self.encoder.image_embedding_size+hdim, 1)

    def forward(self, state, action):
        image_encoded = self.encoder(state)
        action_encoded = F.relu(self.action_encoder(action))
        Q_values = self.decoder(
            torch.cat((image_encoded, action_encoded), dim=-1))
        return Q_values


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