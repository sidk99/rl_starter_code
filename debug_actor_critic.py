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
from a2c import A2C
from agent import Agent
from networks import Policy, ValueFn

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

class Experiment():
    def __init__(self, agent, env, rl_alg):
        self.agent = agent
        self.env = env
        self.rl_alg = rl_alg

    def sample_trajectory(self):
        episode_data = []
        state = self.env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action, log_prob, value = self.agent(state)
            state, reward, done, _ = self.env.step(action)
            if args.render:
                self.env.render()
            mask = 0 if done else 1
            e = {'state': state,
                 'action': action,
                 'logprob': log_prob,
                 'mask': mask,
                 'reward': reward,
                 'value': value}
            episode_data.append(e)
            self.agent.store_transition(e)
            if done:
                break
        returns = sum([e['reward'] for e in episode_data])
        return returns

    def train(self, max_episodes):
        run_avg = RunningAverage()
        for i_episode in range(max_episodes):
            ret = self.sample_trajectory()
            running_return = run_avg.update_variable('reward', ret)
            self.rl_alg.improve(self.agent)
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast Return: {:5d}\tAverage Return: {:.2f}'.format(
                    i_episode, int(ret), running_return))

def main():
    env = gym.make('CartPole-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    agent = Agent(Policy(), ValueFn())
    rl_alg = A2C(gamma=args.gamma)
    experiment = Experiment(agent, env, rl_alg)
    experiment.train(max_episodes=601)


if __name__ == '__main__':
    main()
