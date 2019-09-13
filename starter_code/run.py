import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_algs import PPO, A2C, VPG
from agent import Agent
from policies import DiscretePolicy, SimpleGaussianPolicy, DiscreteCNNPolicy
from value_function import ValueFn, CNNValueFn
import utils
from experiment import Experiment
import gym_minigrid

from log import MultiBaseLogger, MinigridEnvManager, GymEnvManager, create_logdir

import ipdb

from configs import ppo_config, a2c_config, vpg_config

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ppo example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--entropy_coeff', type=float, default=0, metavar='G',
                        help='entropy coeff (default: 0)')
    parser.add_argument('--update-every', type=float, default=1, metavar='G',
                        help='learning rate (default: 1)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--gpu-index', type=int, default=0,
                        help='gpu index (default: 0)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--env-name', type=str, default='InvertedPendulum-v2')
    parser.add_argument('--alg-name', type=str, default='ppo')
    parser.add_argument('--root', type=str, default='runs')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--printf', action='store_true')
    args = parser.parse_args()
    return args

def rlalg_config_switch(alg_name):
    rlalg_configs = {
        'ppo': ppo_config,
        'a2c': a2c_config,
        'vpg': vpg_config
    }
    return rlalg_configs[alg_name]

def rlalg_switch(alg_name):
    rlalgs = {
        'ppo': PPO,
        'a2c': A2C,
        'vpg': VPG
    }
    return rlalgs[alg_name]

def main():
    args = parse_args()
    args = rlalg_config_switch(args.alg_name)(args)
    device=torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if 'MiniGrid' in args.env_name:
        env_manager = MinigridEnvManager(args.env_name, args)
        policy = DiscreteCNNPolicy
        agent = Agent(
            policy(state_dim=env_manager.state_dim, action_dim=env_manager.action_dim), 
            CNNValueFn(state_dim=env_manager.state_dim), args=args).to(device)
    else:
        env_manager = GymEnvManager(args.env_name, args)
        policy = DiscretePolicy if env_manager.is_disc_action else SimpleGaussianPolicy
        agent = Agent(
            policy(state_dim=env_manager.state_dim, hdim=[128, 128], action_dim=env_manager.action_dim), 
            ValueFn(state_dim=env_manager.state_dim), args=args).to(device)

    logger = MultiBaseLogger(args=args)
    env_manager.set_logdir(create_logdir(root=logger.logdir, dirname='{}'.format(args.env_name), setdate=False))
    rl_alg = rlalg_switch(args.alg_name)(device=device, args=args)
    experiment = Experiment(agent, env_manager, rl_alg, logger, device, args)
    experiment.train(max_episodes=100001)


if __name__ == '__main__':
    main()
