import argparse
import os
import torch

from experiment import Experiment
from log import MultiBaseLogger
from rl_algs import rlalg_switch
from run import initialize, create_organism, create_task_progression
import utils as u

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Transfer')
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--ckpt-id', type=int, required=True)

    parser.add_argument('--env-name', type=str, default='InvertedPendulum-v2')
    parser.add_argument('--seed', type=int, default=543)
    parser.add_argument('--alg-name', type=str, default='ppo')
    parser.add_argument('--printf', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args

def load_agent_weights(agent, ckpt, pfunc):
    u.visualize_params({'agent': agent}, pfunc)
    agent.load_state_dict(ckpt['organism'])
    u.visualize_params({'agent': agent}, pfunc)
    return agent

def update_dirs(args, ckpt_subroot, ckpt_expname):
    args.subroot = ckpt_subroot
    args.expname = ckpt_expname + '__to__' + args.expname
    return args

def main():
    args = parse_args()
    args, device = initialize(args)
    ##########################################
    ckpt = torch.load(os.path.join(args.model_dir, 'checkpoints', 'ckpt_batch{}.pth.tar'.format(args.ckpt_id)))
    args = update_dirs(args, ckpt['args'].subroot, ckpt['args'].expname)
    # here assign the task parent here; perhaps have a task-tree object
    ##########################################
    logger = MultiBaseLogger(args=args)
    task_progression = create_task_progression(logger, args)
    agent = create_organism(device, task_progression, args)
    ##########################################
    agent = load_agent_weights(agent, ckpt, logger.printf)
    ##########################################
    rl_alg = rlalg_switch(args.alg_name)(device=device, args=args)
    experiment = Experiment(agent, task_progression, rl_alg, logger, device, args)
    experiment.train(max_epochs=100001)

if __name__ == '__main__':
    main()
    # python starter_code/transfer.py --env-name MiniGrid-Empty-Random-6x6-v0 --model-dir runs/debug/MG-E-R-55-0_s1__2019-09-16_15-19-09 --ckpt-id 300