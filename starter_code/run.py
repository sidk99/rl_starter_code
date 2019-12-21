import argparse
import numpy as np
import torch

from starter_code.agent import Agent
from starter_code.configs import process_config, env_manager_switch
from starter_code.env_config import EnvRegistry as ER
from starter_code.experiment import CentralizedExperiment
from starter_code.log import MultiBaseLogger
from starter_code.policies import DiscretePolicy, SimpleGaussianPolicy, DiscreteCNNPolicy
from starter_code.multitask import construct_task_progression, default_task_prog_spec, task_prog_spec_multi
from starter_code.rb import OnPolicyMemory
from starter_code.rl_algs import rlalg_switch
from starter_code.value_function import ValueFn, CNNValueFn

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Train')
    parser.add_argument('--subroot', type=str, default='debug')
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--env-name', nargs='+', type=str, default=['InvertedPendulum-v2'])
    parser.add_argument('--seed', type=int, default=543)
    parser.add_argument('--alg-name', type=str, default='ppo')
    parser.add_argument('--printf', action='store_true')
    parser.add_argument('--debug', action='store_true')

    # debugging
    parser.add_argument('--max_buffer_size', type=int, default=4096)
    parser.add_argument('--optim_batch_size', type=int, default=256)


    args = parser.parse_args()
    return args

class BaseLauncher:
    env_registry = ER()  # may be mutable?

    @classmethod
    def initialize(cls, args):
        args = process_config(args)
        device=torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() and not args.cpu else torch.device('cpu')
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        return args, device

    @classmethod
    def create_task_progression(cls, logger, args):
        task_progression = construct_task_progression(
                task_prog_spec_multi(args.env_name),
                env_manager_switch(args.env_name[0], cls.env_registry), # first environment of the envs
                logger,
                cls.env_registry,
                args)
        return task_progression

    @classmethod
    def create_organism(cls, device, task_progression, args):
        if 'MiniGrid' in args.env_name[0] or 'BabyAI' in args.env_name[0]:
            policy = DiscreteCNNPolicy(state_dim=task_progression.state_dim, action_dim=task_progression.action_dim)
            critic = CNNValueFn(state_dim=task_progression.state_dim)
        else:
            policy_builder = DiscretePolicy if task_progression.is_disc_action else SimpleGaussianPolicy
            policy = policy_builder(state_dim=task_progression.state_dim, hdim=args.hdim, action_dim=task_progression.action_dim)
            critic = ValueFn(state_dim=task_progression.state_dim)
        replay_buffer = OnPolicyMemory()
        agent = Agent(policy, critic, replay_buffer, args).to(device)
        return agent

    @classmethod
    def main(cls, parse_args):
        args, device = cls.initialize(parse_args())
        logger = MultiBaseLogger(args=args)
        task_progression = cls.create_task_progression(logger, args)
        organism = cls.create_organism(device, task_progression, args)
        rl_alg = rlalg_switch(args.alg_name)(device=device, args=args)
        experiment = CentralizedExperiment(
            agent=organism, 
            task_progression=task_progression, 
            rl_alg=rl_alg,
            logger=logger, 
            device=device, 
            args=args)
        experiment.main_loop(max_epochs=args.max_epochs)

if __name__ == '__main__':
    launcher = BaseLauncher()
    launcher.main(parse_args)