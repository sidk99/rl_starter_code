import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import rlkit.torch.pytorch_util as ptu 
from rlkit.core.serializable import Serializable

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

        self.initialize_optimizer()
        self.initialize_optimizer_schedulers(args)
        print(self)

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
        action_dict = AttrDict(
            action=action,
            stored_action=stored_action,
            action_dist=dist)
        return action_dict

    def update(self, rl_alg):
        rl_alg.improve(self)

    def clear_buffer(self):
        self.replay_buffer.clear_buffer()

    def store_transition(self, transition):
        self.replay_buffer.push(
            transition.state,
            transition.action,
            transition.mask,
            transition.reward,
            )

    def get_state_dict(self):
        state_dict = dict(
            policy=self.policy.state_dict(),
            valuefn=self.valuefn.state_dict(),
            policy_optimizer=self.policy_optimizer.state_dict(),
            value_optimizer=self.value_optimizer.state_dict(),
            )
        return state_dict

    def load_state_dict(self, agent_state_dict, reset_optimizer=True):
        self.policy.load_state_dict(agent_state_dict['policy'])
        self.valuefn.load_state_dict(agent_state_dict['valuefn'])
        if not reset_optimizer:
            self.policy_optimizer.load_state_dict(agent_state_dict['policy_optimizer'])
            self.value_optimizerl.load_state_dict(agent_state_dict['value_optimizer'])

    def get_summary(self):
        return None


class SACAgent(nn.Module):
    def __init__(self, policy, qf1, qf2, target_qf1, target_qf2, replay_buffer, args):
        super(SACAgent, self).__init__()

        # discount=0.99
        # reward_scale=1.0
        # policy_lr=3E-4
        # qf_lr=3E-4
        # optimizer_class = optim.Adam
        # soft_target_tau=5e-3
        # target_update_period=1
        # use_automatic_entropy_tuning=True

        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        # TODO: log_alpha

        self.replay_buffer = replay_buffer
        self.args = args
        assert self.policy.discrete == False
        self.discrete = False

        self.initialize_optimizer()
        self.initialize_optimizer_schedulers(args)
        print(self)

    def initialize_optimizer(self):
        """
            Note that there is an asymmetry here: 
                the alpha optimizer is not listed here.
            TODO: log_alpha
        """
        if self.args.opt == 'adam':
            self.policy_optimizer = optim.Adam(
                self.policy.parameters(), lr=self.args.plr)
            self.qf1_optimizer = optimizer_class(
                self.qf1.parameters(), lr=self.args.vlr)
            self.qf2_optimizer = optimizer_class(
                self.qf2.parameters(), lr=self.args.vlr)
        elif self.args.opt == 'sgd':
            self.policy_optimizer = optim.SGD(
                self.policy.parameters(), lr=self.args.plr, momentum=0.9)
            self.qf1_optimizer = optim.SGD(
                self.qf1.parameters(), lr=self.args.vlr, momentum=0.9)
            self.qf2_optimizer = optim.SGD(
                self.qf2.parameters(), lr=self.args.vlr, momentum=0.9)
        else:
            assert False

    def initialize_optimizer_schedulers(self, args):
        """
            Perhaps you should just anneal everything at the same rate
        """
        if not self.args.anneal_policy_lr: assert self.args.anneal_policy_lr_gamma == 1
        self.po_scheduler = optim.lr_scheduler.StepLR(
            self.policy_optimizer, 
            step_size=args.anneal_policy_lr_step, 
            gamma=args.anneal_policy_lr_gamma, 
            last_epoch=-1)
        self.qf1_scheduler = optim.lr_scheduler.StepLR(
            self.qf1_optimizer, 
            step_size=args.anneal_policy_lr_step, 
            gamma=args.anneal_policy_lr_gamma, 
            last_epoch=-1)
        self.qf2_scheduler = optim.lr_scheduler.StepLR(
            self.qf2_optimizer, 
            step_size=args.anneal_policy_lr_step, 
            gamma=args.anneal_policy_lr_gamma, 
            last_epoch=-1)

    def forward(self, state, deterministic):
        pass

    def update(self, rl_alg):
        rl_alg.improve(self)

    def store_transition(self, transition):
        self.replay_buffer.push(
            transition['state'],
            transition['action'],
            transition['mask'],
            transition['next_state'],
            transition['reward'],
            )

    def get_state_dict(self):
        """
            TODO: need to add in the log_alpha paramter here
        """
        state_dict = dict(
            policy=self.policy.state_dict(),
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy_optimizer=self.policy_optimizer.state_dict(),
            qf1_optimizer=self.qf1_optimizer.state_dict(),
            qf2_optimizer=self.qf2_optimizer.state_dict(),
            )
        raise NotImplementedError('alpha optimizer')
        return state_dict

    def load_state_dict(self, agent_state_dict, reset_optimizer=True):
        self.policy.load_state_dict(agent_state_dict['policy'])
        self.qf1.load_state_dict(agent_state_dict['qf1'])
        self.qf2.load_state_dict(agent_state_dict['qf2'])
        if not reset_optimizer:
            self.policy_optimizer.load_state_dict(agent_state_dict['policy_optimizer'])
            self.qf1_optimizer.load_state_dict(agent_state_dict['qf1_optimizer'])
            self.qf2_optimizer.load_state_dict(agent_state_dict['qf2_optimizer'])

################################################################################


        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True


    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # need to make sure that this matches
        ############################################

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1







