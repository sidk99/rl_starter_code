import argparse
import os
import torch

from experiment import Experiment
from log import MultiBaseLogger
from rl_algs import rlalg_switch
import utils as u

from launcher import BaseLauncher

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Transfer')
    parser.add_argument('--ckpt', type=str, required=True)

    parser.add_argument('--env-name', type=str, default='InvertedPendulum-v2')
    parser.add_argument('--seed', type=int, default=543)
    parser.add_argument('--alg-name', type=str, default='ppo')
    parser.add_argument('--printf', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args

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
        ckpt = torch.load(args.ckpt)
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

if __name__ == '__main__':
    launcher = TransferLauncher()
    launcher.main(parse_args)


# python starter_code/transfer.py --env-name MiniGrid-Empty-Random-6x6-v0 --ckpt runs/debug/MG-E-R-55-0_s1__2019-09-16_15-19-09/checkpoints/ckpt_batch0.pth.tar