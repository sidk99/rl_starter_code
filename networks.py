import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

from utils import normal_log_density, normal_entropy


class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscretePolicy, self).__init__()
        self.action_encoder = nn.Linear(state_dim, 128)
        self.action_head = nn.Linear(128, action_dim)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        action_encoded = F.relu(self.action_encoder(x))
        action_scores = self.action_head(action_encoded)
        action_probs = F.softmax(action_scores, dim=-1)
        return action_probs

    def select_action(self, state, deterministic):
        action_probs = self.forward(state)
        if deterministic:
            action = torch.argmax(action_probs)
        else:
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
        return action

    def get_log_prob(self, state, action):
        action_probs = self.forward(state)
        action_dist = Categorical(action_probs)
        log_prob = action_dist.log_prob(action)
        return log_prob

    def get_entropy(self, state):
        bsize = state.size(0)
        action_probs = self.forward(state)
        action_dist = Categorical(action_probs)
        entropy = action_dist.entropy().view(bsize, 1)
        return entropy

# class ValueFn(nn.Module):
#     def __init__(self, state_dim):
#         super(ValueFn, self).__init__()
#         self.value_affine = nn.Linear(state_dim, 128)
#         self.value_head = nn.Linear(128, 1)

#     def forward(self, state):
#         state_values = self.value_head(F.relu(self.value_affine(state)))
#         return state_values


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


class GaussianParams(nn.Module):
    """
        h --> z
    """
    def __init__(self, hdim, zdim):
        super(GaussianParams, self).__init__()
        self.hdim = hdim
        self.zdim = zdim
        self.mu = nn.Linear(hdim, zdim)
        self.logstd = nn.Linear(hdim, zdim)

        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)

    def forward(self, x):
        mu = self.mu(x)
        logstd = self.logstd(x)
        return mu, logstd

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GaussianPolicy, self).__init__()
        self.encoder1 = nn.Linear(state_dim, 128)
        self.encoder2 = nn.Linear(128, 128)
        self.decoder = GaussianParams(128, action_dim)

    def forward(self, x):
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        mu, logstd = self.decoder(x)
        return mu, torch.exp(logstd)

    def select_action(self, state, deterministic):
        mu, std = self.forward(state)
        if deterministic:
            return mu
        else:
            dist = MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(std))
            action = dist.sample()
            return action

    def get_log_prob(self, state, action):
        mu, std = self.forward(state)
        dist = MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(std))
        log_prob = dist.log_prob(action)#.view(bsize, 1)
        return log_prob


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), activation='relu', log_std=0):
        super().__init__()
        self.is_disc_action = False
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

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x, deterministic):
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}


