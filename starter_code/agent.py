import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from starter_code.rb import OnPolicyMemory

class Agent(nn.Module):
    def __init__(self, policy, valuefn, args):
        super(Agent, self).__init__()
        self.policy = policy
        self.valuefn = valuefn
        self.args = args
        self.discrete = self.policy.discrete

        self.initalize_memory()
        self.initialize_optimizer()
        self.initialize_optimizer_schedulers(args)

        print(self)

    def initalize_memory(self):
        self.buffer = OnPolicyMemory(element='simplertransition')

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

    def step_optimizer_schedulers(self, pfunc):
        def update_optimizer_lr(optimizer, scheduler, name):
            before_lr = optimizer.state_dict()['param_groups'][0]['lr']
            scheduler.step()
            after_lr = optimizer.state_dict()['param_groups'][0]['lr']
            to_print_alr = 'Learning rate for {} was {}. Now it is {}.'.format(name, before_lr, after_lr)
            if before_lr != after_lr:
                to_print_alr += ' Learning rate changed!'
                pfunc(to_print_alr)
        update_optimizer_lr(
            optimizer=self.policy_optimizer,
            scheduler=self.po_scheduler,
            name='policy')
        update_optimizer_lr(
            optimizer=self.value_optimizer,
            scheduler=self.vo_scheduler,
            name='value')

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

    def update(self, rl_alg):
        rl_alg.improve(self)

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

    def load_state_dict(self, agent_state_dict, reset_optimizer=True):
        self.policy.load_state_dict(agent_state_dict['policy'])
        self.valuefn.load_state_dict(agent_state_dict['valuefn'])
        if not reset_optimizer:
            self.policy_optimizer.load_state_dict(agent_state_dict['policy_optimizer'])
            self.value_optimizerl.load_state_dict(agent_state_dict['value_optimizerl'])


class SACAgent(nn.Module):
    pass


