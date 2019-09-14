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

from starter_code.multitask import construct_task_progression, default_task_prog_spec

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Train')
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

    logger = MultiBaseLogger(args=args)

    if 'MiniGrid' in args.env_name:
        task_progression = construct_task_progression(
            default_task_prog_spec(args.env_name),
            MinigridEnvManager, logger, args)
        policy = DiscreteCNNPolicy(state_dim=task_progression.state_dim, action_dim=task_progression.action_dim)
        critic = CNNValueFn(state_dim=task_progression.state_dim)
    else:
        task_progression = construct_task_progression(
            default_task_prog_spec(args.env_name),
            GymEnvManager, logger, args)
        policy_builder = DiscretePolicy if task_progression.is_disc_action else SimpleGaussianPolicy
        policy = policy_builder(state_dim=task_progression.state_dim, hdim=[128, 128], action_dim=task_progression.action_dim)
        critic = ValueFn(state_dim=task_progression.state_dim)

    agent = Agent(policy, critic, args).to(device)
    rl_alg = rlalg_switch(args.alg_name)(device=device, args=args)
    experiment = Experiment(agent, task_progression, rl_alg, logger, device, args)
    experiment.train(max_epochs=100001)

if __name__ == '__main__':
    main()

# python information_economy/scratch/vickrey.py --rlalg ppo --env MiniGrid-Empty-Random-5x5-v0 --envtype mg --policy cbeta --debug --auctiontype bb --critic cnn






