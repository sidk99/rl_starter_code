import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from log import RunningAverage
from rb import Memory
from rl_algs import PPO, A2C, VPG
from agent import Agent
from policies import DiscretePolicy, SimpleGaussianPolicy, DiscreteCNNPolicy
from value_function import ValueFn, CNNValueFn
import utils
from experiment import Experiment
import gym_minigrid

import ipdb

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ppo example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--entropy_coeff', type=float, default=0, metavar='G',
                        help='entropy coeff (default: 0)')
    parser.add_argument('--update-every', type=float, default=1, metavar='G',
                        help='learning rate (default: 200)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--gpu-index', type=int, default=0,
                        help='gpu index (default: 0)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--env-name', type=str, default='InvertedPendulum-v2')
    parser.add_argument('--alg-name', type=str, default='ppo')
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

def ppo_config(args):
    args.plr = 4e-5
    args.vlr = 5e-3
    args.opt = 'sgd'
    args.eval_every = 10
    args.log_every = 1
    return args

def a2c_config(args):
    args.plr = 1e-4
    args.vlr = 5e-3
    args.opt = 'adam'
    args.eval_every = 100
    args.log_every = 10
    return args

def vpg_config(args):
    args.plr = 1e-4
    args.vlr = 5e-3
    args.opt = 'adam'
    args.eval_every = 100
    args.log_every = 10
    return args

def main():
    args = parse_args()
    args = rlalg_config_switch(args.alg_name)(args)
    device=torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

    env = gym.make(args.env_name)

    args.seed = 0
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    if 'MiniGrid' in args.env_name:
        full_state_dim = env.observation_space.spaces['image'].shape  # (H, W, C)
        state_dim = full_state_dim[:-1]  # (H, W)
        is_disc_action = len(env.action_space.shape) == 0
        action_dim = env.action_space.n if is_disc_action else env.action_space.shape[0]

        policy = DiscreteCNNPolicy
        agent = Agent(
            policy(state_dim=state_dim, action_dim=action_dim), 
            CNNValueFn(state_dim=state_dim), args=args).to(device)
    else:
        state_dim = env.observation_space.shape[0]
        is_disc_action = len(env.action_space.shape) == 0
        action_dim = env.action_space.n if is_disc_action else env.action_space.shape[0]

        policy = DiscretePolicy if is_disc_action else SimpleGaussianPolicy
        agent = Agent(
            policy(state_dim=state_dim, hdim=[128, 128], action_dim=action_dim), 
            ValueFn(state_dim=state_dim), args=args).to(device)

    rl_alg = rlalg_switch(args.alg_name)(device=device, args=args)
    experiment = Experiment(agent, env, rl_alg, device, args)
    experiment.train(max_episodes=100001)


if __name__ == '__main__':
    main()
