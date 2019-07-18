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
from rl_algs import PPO
from agent import Agent
from networks import DiscretePolicy, GaussianPolicy, ValueFn, Policy
import utils
from experiment import Experiment

import ipdb

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ppo example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--plr', type=float, default=4e-5, metavar='G',
                        help='learning rate (default: 4e-5)')
    parser.add_argument('--vlr', type=float, default=5e-3, metavar='G',
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--opt', type=str, default='sgd', metavar='G',
                        help='optimizer: adam | sgd (default: sgd')
    parser.add_argument('--update-every', type=float, default=1, metavar='G',
                        help='learning rate (default: 200)')
    parser.add_argument('--eval-every', type=float, default=10, metavar='G',
                        help='eval every (default: 100)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-every', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device=torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

    # env = gym.make('CartPole-v0')
    env = gym.make('InvertedPendulum-v2')
    # env = gym.make('Hopper-v2')
    # env = gym.make('Hopper-v2')

    args.seed = 0
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    is_disc_action = len(env.action_space.shape) == 0
    action_dim = env.action_space.n if is_disc_action else env.action_space.shape[0]
    policy = DiscretePolicy if is_disc_action else GaussianPolicy# Policy

    agent = Agent(
        policy(state_dim=state_dim, action_dim=action_dim), 
        ValueFn(state_dim=state_dim), args=args)
    rl_alg = PPO(device=device, args=args)
    experiment = Experiment(agent, env, rl_alg, args)
    experiment.train(max_episodes=1001)


if __name__ == '__main__':
    main()
