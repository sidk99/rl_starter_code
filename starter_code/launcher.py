import argparse
import itertools
import numpy as np
import os
import torch

from agents import BidAgent
from auction import Vickrey_Auction, Bucket_Brigade
from experiment import GridWorldExperiment, GymExperiment
from starter_code.configs import env_manager_switch, process_config
from starter_code.env_config import EnvRegistry
from starter_code.log import MultiBaseLogger
from starter_code.multitask import construct_task_progression, default_task_prog_spec
from starter_code.policies import BidPolicyLN, DiscretePolicy, SimpleGaussianPolicy, SimpleBetaSoftPlusPolicy, SimpleBetaReluPolicy, BetaCNNPolicy
from starter_code.rl_algs import rlalg_switch
import starter_code.utils as u
from starter_code.value_function import SimpleValueFn, CNNValueFn
from vickrey_log import VickreyLogger

er = EnvRegistry()

def policy_switch(policy_name, state_dim, args):
    policy = {
        'cat': lambda: DiscretePolicy(state_dim, args.hdim, 10),
        'lognormal': lambda: BidPolicyLN(state_dim, args.hdim, 1),
        'beta': lambda: SimpleBetaSoftPlusPolicy(state_dim, args.hdim, 1),
        'betar': lambda: SimpleBetaReluPolicy(state_dim, args.hdim, 1),
        'normal': lambda: SimpleGaussianPolicy(state_dim, args.hdim, 1),
        'cbeta': lambda: BetaCNNPolicy(state_dim, 1)
    }
    return policy[policy_name]

def value_switch(state_dim, args):
    envtype = er.get_env_type(args.env_name)
    if envtype == 'mg':
        value_name = 'cnn'
    else:
        value_name = 'mlp'
    valuefn = {
        'mlp': lambda: SimpleValueFn(state_dim, args.hdim),
        'cnn': lambda: CNNValueFn(state_dim),
    }
    return valuefn[value_name]

def experiment_switch(env_name):
    envtype = er.get_env_type(env_name)
    experiment = {
        'tab': GridWorldExperiment,
        'gym': GymExperiment,
        'mg': GymExperiment,
    }
    return experiment[envtype]

def auction_switch(auctiontype):
    auctiontypes = {
        'v': Vickrey_Auction,
        'bb': Bucket_Brigade,
    }
    return auctiontypes[auctiontype]

def initialize_master_logger(args):
    envtype = er.get_env_type(args.env_name)
    if envtype == 'tab':
        logger = VickreyLogger(args=args)
    elif envtype == 'gym' or envtype == 'mg':
        logger = MultiBaseLogger(args=args)
    else:
        assert False
    return logger

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
    args.action_dim = task_progression.action_dim  # TODO: this should be moved to another method where only the args are modified!
    policy = policy_switch(args.policy, task_progression.state_dim, args)
    valuefn = value_switch(task_progression.state_dim, args)
    agents = [BidAgent(i, policy=policy(),valuefn=valuefn(),args=args).to(device) for i in range(task_progression.action_dim * len(args.parents))]
    organism = auction_switch(args.auctiontype)(agents=agents, device=device, args=args)
    return organism

def load_agent_weights(agent, ckpts, pfunc):
    u.visualize_params({'agent': agent}, pfunc)
    society_state_dict = list(itertools.chain.from_iterable([c['organism'] for c in ckpts]))
    agent.load_state_dict(society_state_dict)
    u.visualize_params({'agent': agent}, pfunc)
    return agent

def update_dirs(args, ckpt_expnames):
    args.expname = '__and__'.join(ckpt_expnames) + '__to__' + args.expname
    return args

def transfer_args(args, ckpts):
    assert u.all_same([c['args'].policy for c in ckpts])  # in general this need not be the same
    assert u.all_same([c['args'].auctiontype for c in ckpts])
    assert u.all_same([c['args'].ado for c in ckpts])
    args.policy = ckpts[0]['args'].policy
    args.auctiontype = ckpts[0]['args'].auctiontype  # this should be the same I think
    args.ado = ckpts[0]['args'].ado  # this should be the same too
    args.parents = args.ckpts
    return args

def main():
    args, device = initialize(parse_args())
    ##########################################
    ckpts = [torch.load(c) for c in args.ckpts]
    args = update_dirs(args, [c['args'].expname for c in ckpts])
    # here assign the task parent here; perhaps have a task-tree object
    args = transfer_args(args, ckpts)
    ##########################################
    logger = initialize_master_logger(args)
    task_progression = create_task_progression(logger, args)
    organism = create_organism(device, task_progression, args)
    ##########################################
    organism = load_agent_weights(organism, ckpts, logger.printf)
    ##########################################
    rl_alg = rlalg_switch(args.alg_name)(device=device, args=args)
    experiment = experiment_switch(args.env_name)(
        society=organism, rl_alg=rl_alg, logger=logger, args=args, task_progression=task_progression, device=device)
    experiment.train(max_epochs=args.max_epochs)


if __name__ == '__main__':
    main()

# python information_economy/scratch/vickrey_transfer.py --env-name MiniGrid-Empty-Random-6x6-v0 --ckpt runs/debug/MG-E-R-55-0_s0_db__2019-09-16_23-19-22/checkpoints/ckpt_batch18.pth.tar runs/debug/MG-E-R-55-0_s0_db__2019-09-16_23-19-22/checkpoints/ckpt_batch12.pth.tar --debug --subroot debug

# python information_economy/scratch/vickrey_bucket_transfer.py --env-name MiniGrid-Empty-Random-6x6-v0 --ckpts runs/debug/MG-E-R-55-0_s0_db__2019-09-16_23-19-22/checkpoints/ckpt_batch18.pth.tar --debug --subroot debug


