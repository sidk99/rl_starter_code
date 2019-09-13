import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from rb import Memory

class Agent(nn.Module):
    def __init__(self, policy, valuefn, args):
        super(Agent, self).__init__()
        self.policy = policy
        self.valuefn = valuefn
        self.args = args

        self.initalize_memory()
        self.initialize_optimizer()
        self.initialize_optimizer_schedulers(args)

        print(self)

    def initalize_memory(self):
        self.buffer = Memory(element='simplertransition')

    def initialize_optimizer(self):
        if self.args.opt == 'adam':
            self.policy_optimizer = optim.Adam(
                self.policy.parameters(), lr=self.args.plr)
            self.value_optimizer = optim.Adam(
                self.valuefn.parameters(), lr=self.args.vlr)
        elif self.args.opt == 'sgd':
            self.policy_optimizer = optim.SGD(
                self.policy.parameters(), lr=self.args.plr, momentum=0.9)
            self.value_optimizer = optim.SGD(
                self.valuefn.parameters(), lr=self.args.vlr, momentum=0.9)
        else:
            assert False

    def initialize_optimizer_schedulers(self, args):
        if not self.args.anneal_policy_lr: assert self.args.anneal_policy_lr_gamma == 1
        self.po_scheduler = optim.lr_scheduler.StepLR(
            self.policy_optimizer, 
            step_size=args.anneal_policy_lr_step, 
            gamma=args.anneal_policy_lr_gamma, 
            last_epoch=-1)
        self.vo_scheduler = optim.lr_scheduler.StepLR(
            self.value_optimizer, 
            step_size=args.anneal_policy_lr_step, 
            gamma=args.anneal_policy_lr_gamma, 
            last_epoch=-1)

    def forward(self, state, deterministic):
        action, dist = self.policy.select_action(state, deterministic)
        action = action.detach()[0].cpu().numpy()  # (adim)
        if self.policy.discrete:
            action = int(action)
            stored_action = [action]
        else:
            stored_action = action
        action_dict = {
            'action': action, 
            'stored_action': stored_action, 
            'action_dist': dist}
        return action_dict

    def store_transition(self, transition):
        self.buffer.push(
            transition['state'],
            transition['action'],
            transition['mask'],
            transition['reward'],
            )

    def get_state_dict(self):
        state_dict = {
            'policy': self.policy.state_dict(),
            'valuefn': self.valuefn.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            }
        return state_dict
