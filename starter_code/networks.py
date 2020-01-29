import torch
import torch.nn as nn
import torch.nn.functional as F

from starter_code.utils import normal_log_density, normal_entropy


class CNN(nn.Module):
    # from rl-starter-files
    def __init__(self, n, m):
        super(CNN, self).__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
    def forward(self, x):
        # (bsize, H, W, C) --> (bsize, C, H, W)
        x = x.transpose(1, 3).transpose(2, 3).contiguous()
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        return x

class MLP(nn.Module):
    def __init__(self, dims, zero_init=False, output_activation=None):
        super(MLP, self).__init__()
        assert len(dims) >= 2
        self.network = nn.ModuleList([])
        for i in range(len(dims)-1):
            layer = nn.Linear(dims[i], dims[i+1])
            if zero_init:
                layer.weight.data.zero_()
                layer.bias.data.zero_()
            self.network.append(layer)
        self.output_activation = output_activation

    def forward(self, x):
        for i, layer in enumerate(self.network):
            x = layer(x)
            if i < len(self.network)-1:
                x = F.relu(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x

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

class BetaSoftPlusParams(nn.Module):
    """
        h --> z
    """
    def __init__(self, hdim, zdim):
        super(BetaSoftPlusParams, self).__init__()
        self.hdim = hdim
        self.zdim = zdim
        self.alpha = nn.Linear(hdim, zdim)
        self.beta = nn.Linear(hdim, zdim)

        # initialize to uniform
        self.alpha.weight.data.fill_(0.0)
        self.alpha.bias.data.fill_(1.0)

        self.beta.weight.data.fill_(0.0)
        self.beta.bias.data.fill_(1.0)

    def forward(self, x):
        alpha = F.softplus(self.alpha(x))  # the parameters for softplus are tunable
        beta = F.softplus(self.beta(x))  # the parameters for softplus are tunable
        return alpha, beta


class BetaReluParams(nn.Module):
    """
        h --> z
    """
    def __init__(self, hdim, zdim):
        super(BetaReluParams, self).__init__()
        self.hdim = hdim
        self.zdim = zdim
        self.alpha = nn.Linear(hdim, zdim)
        self.beta = nn.Linear(hdim, zdim)

        # initialize to uniform
        self.alpha.weight.data.fill_(0.0)
        self.alpha.bias.data.fill_(0.0)

        self.beta.weight.data.fill_(0.0)
        self.beta.bias.data.fill_(3.0)

    def forward(self, x):    
        alpha = F.relu(self.alpha(x)) + 1
        beta = F.relu(self.beta(x)) + 1
        return alpha, beta