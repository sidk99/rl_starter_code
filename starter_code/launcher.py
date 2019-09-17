import argparse
import numpy as np
import os
import torch

from agent import Agent
from configs import process_config, env_manager_switch
from experiment import Experiment
from log import MultiBaseLogger
from policies import DiscretePolicy, SimpleGaussianPolicy, DiscreteCNNPolicy
from starter_code.multitask import construct_task_progression, default_task_prog_spec
from starter_code.rl_algs import rlalg_switch
import utils as u
from value_function import ValueFn, CNNValueFn

class BaseLauncher:

    @classmethod
    def initialize(cls, args):
        args = process_config(args)
        device=torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        return args, device

    @classmethod
    def create_task_progression(cls, logger, args):
        task_progression = construct_task_progression(
                default_task_prog_spec(args.env_name),
                env_manager_switch(args.env_name), logger, args)
        return task_progression

    @classmethod
    def create_organism(cls, device, task_progression, args):
        if 'MiniGrid' in args.env_name:
            policy = DiscreteCNNPolicy(state_dim=task_progression.state_dim, action_dim=task_progression.action_dim)
            critic = CNNValueFn(state_dim=task_progression.state_dim)
        else:
            policy_builder = DiscretePolicy if task_progression.is_disc_action else SimpleGaussianPolicy
            policy = policy_builder(state_dim=task_progression.state_dim, hdim=args.hdim, action_dim=task_progression.action_dim)
            critic = ValueFn(state_dim=task_progression.state_dim)
        agent = Agent(policy, critic, args).to(device)
        return agent

    @classmethod
    def main(cls, parse_args):
        args, device = cls.initialize(parse_args())
        logger = MultiBaseLogger(args=args)
        task_progression = cls.create_task_progression(logger, args)
        organism = cls.create_organism(device, task_progression, args)
        rl_alg = rlalg_switch(args.alg_name)(device=device, args=args)
        experiment = Experiment(organism, task_progression, rl_alg, logger, device, args)
        experiment.train(max_epochs=args.max_epochs)


class TransferLauncher(BaseLauncher):

    @classmethod
    def load_agent_weights(cls, agent, ckpt, pfunc):
        u.visualize_params({'agent': agent}, pfunc)
        agent.load_state_dict(ckpt['organism'])
        u.visualize_params({'agent': agent}, pfunc)
        return agent

    @classmethod
    def update_dirs(cls, args, ckpt_subroot, ckpt_expname):
        args.subroot = ckpt_subroot
        args.expname = ckpt_expname + '__to__' + args.expname
        return args

    @classmethod
    def main(cls, parse_args):
        args, device = cls.initialize(parse_args())
        ##########################################
        ckpt = torch.load(os.path.join(args.model_dir, 'checkpoints', 'ckpt_batch{}.pth.tar'.format(args.ckpt_id)))
        args = cls.update_dirs(args, ckpt['args'].subroot, ckpt['args'].expname)
        # here assign the task parent here; perhaps have a task-tree object
        ##########################################
        logger = MultiBaseLogger(args=args)
        task_progression = cls.create_task_progression(logger, args)
        agent = cls.create_organism(device, task_progression, args)
        ##########################################
        agent = cls.load_agent_weights(agent, ckpt, logger.printf)
        ##########################################
        rl_alg = rlalg_switch(args.alg_name)(device=device, args=args)
        experiment = Experiment(agent, task_progression, rl_alg, logger, device, args)
        experiment.train(max_epochs=100001)

# maybe these should be classmethods actually



