import argparse
import gym
import numpy as np
import os
import sys
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from log import RunningAverage
from rb import Memory
from ppo import PPO
from agent import BaseActionAgent

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--lr', type=float, default=3e-2, metavar='R',
                    help='learning rate (default: 3e-2)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='I',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
device=torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

eps = np.finfo(np.float32).eps.item()

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.action_affine = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)

    def forward(self, state):
        action_scores = self.action_head(F.relu(self.action_affine(state)))
        action_dist = F.softmax(action_scores, dim=-1)
        return action_dist

    def select_action(self, state):
        # volatile
        action_dist = self.forward(state)
        m = Categorical(action_dist)
        action = m.sample()
        return action.data

    def get_log_prob(self, state, action):
        # not volatile
        action_dist = self.forward(state)
        m = Categorical(action_dist)
        log_prob = m.log_prob(action)
        return log_prob

class ValueFn(nn.Module):
    def __init__(self):
        super(ValueFn, self).__init__()
        self.value_affine = nn.Linear(4, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        state_values = self.value_head(F.relu(self.value_affine(state)))
        return state_values

agent = BaseActionAgent(policy=Policy(), valuefn=ValueFn(), id=0, device=device, args=args)

def sample_trajectory(agent, env):
    episode_data = []
    state = env.reset()
    for t in range(10000):  # Don't infinite loop while learning
        action, log_prob, value = agent(torch.from_numpy(state).float() )
        state, reward, done, _ = env.step(action)
        if args.render:
            env.render()
        mask = 0 if done else 1
        e = {'state': state, 
             'action': action,
             'logprob': log_prob,
             'mask': mask,
             'reward': reward,
             'value': value}
        episode_data.append(e)
        agent.store_transition(e)
        if done:
            break
    returns = sum([e['reward'] for e in episode_data])
    return returns, t

def main():
    run_avg = RunningAverage()
    for i_episode in range(1, 501):
        ret, t = sample_trajectory(agent, env)
        running_reward = run_avg.update_variable('reward', ret)
        if i_episode % 50 == 0:
            agent.improve()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
