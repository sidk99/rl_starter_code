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
import utils as u
from experiment import Experiment
import gym_minigrid

from log import MultiBaseLogger, MinigridEnvManager, GymEnvManager, create_logdir

import ipdb
import os


from configs import ppo_config, a2c_config, vpg_config

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Transfer')


    parser.add_argument('--model_dir', type=str, default=None,
                    help='Directory containing model to load')
    parser.add_argument('--ckpt_id', type=int, default=0,
                    help='Checkpoint id to load from') 
    parser.add_argument('--env-name', type=str, default='InvertedPendulum-v2')
    parser.add_argument('--expname', type=str, default='')


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
    parser.add_argument('--alg-name', type=str, default='ppo')
    parser.add_argument('--root', type=str, default='runs')
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

def load_checkpoint(agent, args):
    # load checkpoint
    u.visualize_params({'agent': agent}, print)
    checkpoint = torch.load(os.path.join(args.model_dir, 'checkpoints', 'ckpt_batch{}.pth.tar'.format(args.ckpt_id)))
    agent.load_state_dict(checkpoint['organism'])
    u.visualize_params({'agent': agent}, print)
    return agent

def main():
    args = parse_args()
    args = rlalg_config_switch(args.alg_name)(args)
    device=torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if 'MiniGrid' in args.env_name:
        env_manager = MinigridEnvManager(args.env_name, args)
        policy = DiscreteCNNPolicy(state_dim=env_manager.state_dim, action_dim=env_manager.action_dim)
        critic = CNNValueFn(state_dim=env_manager.state_dim)
    else:
        env_manager = GymEnvManager(args.env_name, args)
        policy_builder = DiscretePolicy if env_manager.is_disc_action else SimpleGaussianPolicy
        policy = policy_builder(state_dim=env_manager.state_dim, hdim=[128, 128], action_dim=env_manager.action_dim)
        critic = ValueFn(state_dim=env_manager.state_dim)

    agent = Agent(policy, critic, args).to(device)
    agent = load_checkpoint(agent, args)

    logger = MultiBaseLogger(args=args)
    env_manager.set_logdir(create_logdir(root=logger.logdir, dirname='{}'.format(args.env_name), setdate=False))
    rl_alg = rlalg_switch(args.alg_name)(device=device, args=args)


    experiment = Experiment(agent, env_manager, rl_alg, logger, device, args)
    experiment.train(max_episodes=100001)

if __name__ == '__main__':
    main()



    # python starter_code/run.py --env-name MiniGrid-Empty-Random-5x5-v0 --seed 1 --expname empty_mg_test
    # python starter_code/transfer.py --env-name MiniGrid-Empty-Random-6x6-v0 --expname empty_mg_test_transfer --model_dir <blah> --ckpt_id <>




