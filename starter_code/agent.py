import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# import rlkit.torch.pytorch_util as ptu 
# from rlkit.core.serializable import Serializable

from starter_code.utils import AttrDict

class Agent(nn.Module):
    """
        args.opt
        args.plr
        args.vlr
        args.anneal_policy_lr
        args.anneal_policy_lr_gamma
    """
    def __init__(self, policy, valuefn, replay_buffer, args):
        super(Agent, self).__init__()
        self.policy = policy
        self.valuefn = valuefn
        self.replay_buffer = replay_buffer
        self.args = args
        self.discrete = self.policy.discrete

        self.bundle_networks()
        self.initialize_optimizer(lrs=dict(policy=self.args.plr, valuefn=self.args.vlr))
        self.initialize_optimizer_schedulers(args)

    def bundle_networks(self):
        self.networks = dict(
            policy=self.policy, 
            valuefn=self.valuefn)  

    def initialize_optimizer(self, lrs):
        self.optimizers = {}
        for name, network in self.networks.items():
            if self.args.opt == 'adam':
                self.optimizers[name] = optim.Adam(
                    network.parameters(), lr=lrs[name])
            elif self.args.opt == 'sgd':
                self.optimizers[name] = optim.SGD(
                    network.parameters(), lr=lrs[name], momentum=0.9)
            else:
                assert False

    def initialize_optimizer_schedulers(self, args):
        if not self.args.anneal_policy_lr: assert self.args.anneal_policy_lr_gamma == 1
        self.schedulers = {}
        for name, optimizer in self.optimizers.items():
            self.schedulers[name] = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=args.anneal_policy_lr_step, 
                gamma=args.anneal_policy_lr_gamma,
                last_epoch=-1)

    def step_optimizer_schedulers(self, pfunc):
        verbose = False
        for name in self.schedulers:
            before_lr = self.optimizers[name].state_dict()['param_groups'][0]['lr']
            self.schedulers[name].step()
            after_lr = self.optimizers[name].state_dict()['param_groups'][0]['lr']
            if verbose:
                to_print_alr = 'Learning rate for {} was {}. Now it is {}.'.format(name, before_lr, after_lr)
                if before_lr != after_lr:
                    to_print_alr += ' Learning rate changed!'
                    pfunc(to_print_alr)

    def forward(self, state, deterministic):
        action, dist = self.policy.select_action(state, deterministic)
        action = action.detach()[0].cpu().numpy()  # (adim)
        if self.policy.discrete:
            action = int(action)
            stored_action = [action]
        else:
            stored_action = action
        action_dict = dict(
            action=action,
            stored_action=stored_action,
            action_dist=dist
            )
        return action_dict

    def update(self, rl_alg):
        rl_alg.improve(self)

    def clear_buffer(self):
        self.replay_buffer.clear_buffer()

    def store_transition(self, transition):
        self.replay_buffer.push(
            state=transition.state,
            action=transition.action,
            next_state=transition.next_state,
            mask=transition.mask,
            reward=transition.reward,
            )

    def get_state_dict(self):
        state_dict = dict()
        for name in self.networks:
            state_dict[name] = self.networks[name].state_dict()
        for name in self.optimizers:
            state_dict['{}_optimizer'.format(name)] = self.optimizers[name].state_dict()
        return state_dict

    def load_state_dict(self, agent_state_dict, reset_optimizer=True):
        for name in self.networks:
            self.networks[name].load_state_dict(agent_state_dict[name])
        if not reset_optimizer:
            for name in self.optimizers:
                self.optimizers[name].load_state_dict(
                    agent_state_dict['{}_optimizer'.format(name)])  # TODO: should make this a subdictionary

    def get_summary(self):
        return None


# reward_scale=1.0
# policy_lr=3E-4
# qf_lr=3E-4
# soft_target_tau=5e-3
# target_update_period=1
# use_automatic_entropy_tuning=True
class SACAgent(Agent):
    def __init__(self, policy, qf1, qf2, target_qf1, target_qf2, replay_buffer, args):
        nn.Module.__init__(self)
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        if args.use_automatic_entropy_tuning:  # TODO
            self.log_alpha = ptu.zeros(1, requires_grad=True)  # TODO

        self.replay_buffer = replay_buffer
        self.args = args
        assert self.policy.discrete == False  # TODO: can SAC only work for continous actions?
        self.discrete = False

        self.bundle_networks()
        self.initialize_optimizer(lrs=dict(
            policy=self.args.plr,
            qf1=self.args.vlr,
            qf2=self.args.vlr))
        self.initialize_optimizer_schedulers(args)

    def bundle_networks(self):
        # this is just for the state_dict
        self.networks = dict(
            policy=self.policy, 
            qf1=self.qf1,
            qf2=self.qf2)
        # note that I am not including self.log_alpha here!

    def initialize_optimizer(self, lrs):
        Agent.initialize_optimizer(self, lrs)
        if args.use_automatic_entropy_tuning:
            if self.args.opt == 'adam':
                self.optimizers['alpha_optimizer'] = optim.Adam(
                    [self.log_alpha], lr=self.args.plr)
            elif self.args.opt == 'sgd':
                self.optimizers['alpha_optimizer'] = optim.SGD(
                    [self.log_alpha], lr=self.args.plr, momentum=0.9)
            else:
                assert False

    # initialize_optimizer_schedulers() inherited
    # step_optimizer_schedulers() inherited
    # forward() inherited
    # update() inherited
    # clear_buffer() inherited
    # store_transition() inherited

    def get_state_dict(self):
        """
            TODO: need to add in the log_alpha paramter here
        """
        state_dict = Agent.get_state_dict(self)
        if args.use_automatic_entropy_tuning:
            state_dict['log_alpha'] = self.log_alpha.item()  # TODO: figure out if you should detach
        return state_dict

    def load_state_dict(self, agent_state_dict, reset_optimizer=True):
        Agent.load_state_dict(
            agent_state_dict=agent_state_dict, 
            reset_optimizer=reset_optimizer)
        self.log_alpha = agent_state_dict['log_alpha']  # should I convert to tensor?








