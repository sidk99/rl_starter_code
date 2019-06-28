import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from log import RunningAverage
from rb import Memory
from a2c import A2C

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

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.action_encoder = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        action_encoded = F.relu(self.action_encoder(x))
        action_scores = self.action_head(action_encoded)
        return F.softmax(action_scores, dim=-1)#, state_values


class ValueFn(nn.Module):
    def __init__(self):
        super(ValueFn, self).__init__()
        self.value_affine = nn.Linear(4, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        state_values = self.value_head(F.relu(self.value_affine(state)))
        return state_values


class Agent(nn.Module):
    def __init__(self, policy, valuefn):
        super(Agent, self).__init__()
        self.policy = policy
        self.valuefn = valuefn

        self.initalize_memory()
        self.initialize_optimizer()
        # self.initialize_rl_alg()

    def initalize_memory(self):
        self.buffer = Memory(element='simpletransition')

    def initialize_optimizer(self):
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.valuefn.parameters(), lr=3e-4)

    def forward(self, x):
        probs, state_value = self.policy(x), self.valuefn(x)
        return probs, state_value

    def forward(self, state):
        state = torch.from_numpy(state).float()
        action_dist = self.policy(state)
        value = self.valuefn(state)
        m = Categorical(action_dist)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob, value

    def store_transition(self, transition):
        self.buffer.push(
            transition['state'],
            transition['action'],
            transition['logprob'],
            transition['mask'],
            transition['reward'],
            transition['value'])

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

    def train(self):
        run_avg = RunningAverage()
        for i_episode in range(601):
            ret = self.sample_trajectory()
            running_reward = run_avg.update_variable('reward', ret)
            self.rl_alg.improve(self.agent)
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                    i_episode, int(ret), running_reward))
            if running_reward > self.env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, int(ret)))
                break

def main():
    env = gym.make('CartPole-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    agent = Agent(Policy(), ValueFn())
    rl_alg = A2C(gamma=args.gamma)
    experiment = Experiment(agent, env, rl_alg)
    experiment.train()


if __name__ == '__main__':
    main()
