import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from starter_code.rl_algs import PPO, A2C, VPG, rlalg_switch
from agent import Agent
from policies import DiscretePolicy, SimpleGaussianPolicy, DiscreteCNNPolicy
from value_function import ValueFn, CNNValueFn
import utils
from experiment import Experiment
import gym_minigrid

from log import MultiBaseLogger, MinigridEnvManager, GymEnvManager, create_logdir

import ipdb

from configs import process_config, env_manager_switch

from starter_code.multitask import construct_task_progression, default_task_prog_spec


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Train')
    parser.add_argument('--subroot', type=str, default='debug')

    parser.add_argument('--env-name', type=str, default='InvertedPendulum-v2')
    parser.add_argument('--seed', type=int, default=543)
    parser.add_argument('--alg-name', type=str, default='ppo')
    parser.add_argument('--printf', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args

def initialize(args):
    args = process_config(args)
    device=torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args, device

def create_task_progression(logger, args):
    task_progression = construct_task_progression(
            default_task_prog_spec(args.env_name),
            env_manager_switch(args.env_name), logger, args)
    return task_progression

def create_organism(device, task_progression, args):
    if 'MiniGrid' in args.env_name:
        policy = DiscreteCNNPolicy(state_dim=task_progression.state_dim, action_dim=task_progression.action_dim)
        critic = CNNValueFn(state_dim=task_progression.state_dim)
    else:
        policy_builder = DiscretePolicy if task_progression.is_disc_action else SimpleGaussianPolicy
        policy = policy_builder(state_dim=task_progression.state_dim, hdim=args.hdim, action_dim=task_progression.action_dim)
        critic = ValueFn(state_dim=task_progression.state_dim)
    agent = Agent(policy, critic, args).to(device)
    return agent

def main():
    args = parse_args()
    args, device = initialize(args)
    logger = MultiBaseLogger(args=args)
    task_progression = create_task_progression(logger, args)
    agent = create_organism(device, task_progression, args)
    rl_alg = rlalg_switch(args.alg_name)(device=device, args=args)
    experiment = Experiment(agent, task_progression, rl_alg, logger, device, args)
    experiment.train(max_epochs=args.max_epochs)

if __name__ == '__main__':
    main()

    # python starter_code/run.py --env-name MiniGrid-Empty-Random-5x5-v0 --seed 1

