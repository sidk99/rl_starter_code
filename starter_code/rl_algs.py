import copy
import math
import numpy as np
import pickle
import sys
import torch
import torch.nn.functional as F

from collections import defaultdict

import rlkit.torch.pytorch_util as ptu
from starter_code.common import estimate_advantages
import starter_code.utils as u

import time
import gtimer as gt

import rlkit.torch.pytorch_util as ptu 


eps = np.finfo(np.float32).eps.item()

def analyze_size(obj, obj_name):
    obj_pickle = pickle.dumps(obj)
    print('Size of {}: {}'.format(obj_name, sys.getsizeof(obj_pickle)))

def rlalg_switch(alg_name):
    rlalgs = {
        'ppo': PPO,
        'a2c': A2C,
        'vpg': VPG,
        'sac': SAC,
    }
    return rlalgs[alg_name]


class OnPolicyRLAlg():
    def __init__(self, device, max_buffer_size):
        self.device = device
        self.max_buffer_size = max_buffer_size
        self.num_samples_before_update = max_buffer_size

class OffPolicyRlAlg():
    def __init__(self, device, max_buffer_size, num_samples_before_update):
        self.device = device
        self.max_buffer_size = max_buffer_size
        self.num_samples_before_update = num_samples_before_update
        assert self.num_samples_before_update <= self.max_buffer_size


class VPG(OnPolicyRLAlg):
    """
        args
    """
    def __init__(self, device, args):
        super(VPG, self).__init__(device=device, max_buffer_size=args.max_buffer_size)
        self.gamma = self.args.gamma

    def unpack_batch(self, batch):
        states = torch.from_numpy(np.stack(batch.state)).to(torch.float32).to(self.device)  # (bsize, sdim)
        actions = torch.from_numpy(np.stack(batch.action)).to(torch.float32).to(self.device)  # (bsize, adim)
        assert actions.dim() == 2 and (states.dim() == 2 or states.dim() == 4)
        return states, actions

    def improve(self, agent):
        batch = agent.replay_buffer.sample()
        states, actions = self.unpack_batch(batch)
        log_probs = agent.policy.get_log_prob(states, actions)
        assert log_probs.dim() == 2

        R = 0
        policy_losses = []
        rewards = []
        for r in batch.reward[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, reward in zip(log_probs, rewards):
            policy_losses.append(-log_prob * reward)
        agent.optimizers['policy'].zero_grad()
        loss = torch.stack(policy_losses).sum()
        loss.backward()
        agent.optimizers['policy'].step()
        agent.replay_buffer.clear_buffer()


class A2C(OnPolicyRLAlg):
    """
        args
    """
    def __init__(self, device, args):
        super(A2C, self).__init__(device=device, max_buffer_size=args.max_buffer_size)
        self.gamma = self.args.gamma

    def unpack_batch(self, batch):
        states = torch.from_numpy(np.stack(batch.state)).to(torch.float32).to(self.device)  # (bsize, sdim)
        actions = torch.from_numpy(np.stack(batch.action)).to(torch.float32).to(self.device)  # (bsize, adim)
        assert actions.dim() == 2 and (states.dim() == 2 or states.dim() == 4)
        return states, actions

    def improve(self, agent):
        batch = agent.replay_buffer.sample()
        states, actions = self.unpack_batch(batch)
        values = agent.valuefn(states)
        log_probs = agent.policy.get_log_prob(states, actions)
        assert values.dim() == log_probs.dim() == 2

        R = 0
        policy_losses = []
        value_losses = []
        rewards = []
        for r in batch.reward[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, value, r in zip(log_probs, values, rewards):
            reward = r - value.item()
            policy_losses.append(-log_prob * reward)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([r]).to(self.device)))
        agent.optimizers['policy'].zero_grad()
        agent.optimizers['valuefn'].zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        agent.optimizers['policy'].step()
        agent.optimizers['valuefn'].step()
        agent.replay_buffer.clear_buffer()


class PPO(OnPolicyRLAlg):
    """
        args.entropy_coeff
        args.plr (will remove)
    """
    def __init__(self, device, args):
        super(PPO, self).__init__(device=device, max_buffer_size=args.max_buffer_size)
        self.device = device
        self.args = args

        # self.gamma = 0.99
        self.gamma = self.args.gamma
        self.tau = 0.95
        self.l2_reg = 1e-3
        self.clip_epsilon = 0.2
        self.entropy_coeff = args.entropy_coeff

        self.optim_epochs = 1#5#10  # this may be why it takes so long
        self.optim_batch_size = args.optim_batch_size
        self.optim_value_iternum = 1

        self.reset_record()

    def record(self, minibatch_log, epoch, iter):
        self.log[epoch][iter] = minibatch_log

    def reset_record(self):
        self.log = defaultdict(dict)

    def aggregate_stats(self):
        stats = defaultdict(dict)
        aggregators = {'avg': np.mean, 'max': np.max, 'min': np.min, 'std': np.std}
        metrics = copy.deepcopy(list(self.log[0][0].keys()))
        metrics.remove('bsize')
        for m in metrics:
            metric_data = []
            for e in self.log:
                epoch_metric_data = [v[m] for k, v in self.log[e].items()]
                metric_data.extend(epoch_metric_data)
            for a in aggregators:
                stats[m][a] = aggregators[a](metric_data)
        return stats

    def unpack_batch(self, batch):
        time_start_batch = time.time()
        states = torch.from_numpy(np.stack(batch.state)).to(torch.float32).to(self.device)  # (bsize, sdim)
        actions = torch.from_numpy(np.stack(batch.action)).to(torch.float32).to(self.device)  # (bsize, adim)
        masks = torch.from_numpy(np.stack(batch.mask)).to(torch.float32).to(self.device)  # (bsize)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(torch.float32).to(self.device)  # (bsize)
        assert actions.dim() == 2 and (states.dim() == 2 or states.dim() == 4)
        assert masks.dim() == rewards.dim() == 1
        return states, actions, masks, rewards

    def improve(self, agent):
        self.reset_record()
        batch = agent.replay_buffer.sample()
        states, actions, masks, rewards = self.unpack_batch(batch)

        with torch.no_grad():
            values = agent.valuefn(states)  # (bsize, 1)
            fixed_log_probs = agent.policy.get_log_prob(
                agent.policy.get_action_dist(states), actions)  # (bsize, 1)
            assert values.dim() == fixed_log_probs.dim() == 2

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau, self.device)  # (bsize, 1) (bsize, 1)

        """perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(states.shape[0] / self.optim_batch_size))

        for j in range(self.optim_epochs):
            time_before_permute = time.time()
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)

            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                minibatch_log = self.ppo_step(agent, states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b)
                self.record(minibatch_log=minibatch_log, epoch=j, iter=i)

        agent.replay_buffer.clear_buffer()

    def ppo_step(self, agent, states, actions, returns, advantages, fixed_log_probs):
        """
            states: (minibatch_size, H, W, C)
            actions: 
            returns:
            advantages: 
            fixed_log_probs:

            entropy: (minibatch_size, 1)
        """
        """update critic"""
        for _ in range(self.optim_value_iternum):
            values_pred = agent.valuefn(states)
            value_loss = (values_pred - returns).pow(2).mean()
            # weight decay
            for param in agent.valuefn.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg
            agent.optimizers['valuefn'].zero_grad()
            value_loss.backward()
            agent.optimizers['valuefn'].step()

        """update policy"""
        action_dist = agent.policy.get_action_dist(states)
        log_probs = agent.policy.get_log_prob(action_dist, actions)  # (bsize, 1)
        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_surr = -torch.min(surr1, surr2).mean()
        entropy = agent.policy.get_entropy(action_dist).mean()

        policy_loss = policy_surr - self.entropy_coeff*entropy
        agent.optimizers['policy'].zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 40)
        agent.optimizers['policy'].step()

        """log"""
        num_clipped = (surr1-surr2).nonzero().size(0)
        ratio_clipped = num_clipped / states.size(0)
        log = {}
        log['num_clipped'] = num_clipped
        log['ratio_clipped'] = ratio_clipped
        log['entropy'] = entropy
        log['bsize'] = states.size(0)
        log['value_loss'] = value_loss.item()
        log['policy_surr'] = policy_surr.item()
        log['policy_loss'] = policy_loss.item()
        return log


"""
Note that these are the args:


  "algorithm": "SAC",
  "version": "normal",
  "layer_size": 256,
  "replay_buffer_size": 1000000,
  "algorithm_kwargs": {
    "num_epochs": 3000,
    "num_eval_steps_per_epoch": 5000,
    "num_trains_per_train_loop": 1000,
    "num_expl_steps_per_train_loop": 1000,
    "min_num_steps_before_training": 1000,
    "max_path_length": 1000,
    "batch_size": 256
  },
  "trainer_kwargs": {
    "discount": 0.99,
    "soft_target_tau": 0.005,
    "target_update_period": 1,
    "policy_lr": 0.0003,
    "qf_lr": 0.0003,
    "reward_scale": 1,
    "use_automatic_entropy_tuning": true
  }
}




"""


class SAC(OffPolicyRlAlg):
    """
        TODO:
            * self.alpha_optimizer
            * make sure select_action can be done in batch
    """
    def __init__(self, device, args):
        super(SAC, self).__init__(device, 
            max_buffer_size=args.max_buffer_size,
            num_samples_before_update=args.num_samples_before_update)
        self.args = args
        self.gamma = self.args.gamma

        self.optim_batch_size = self.args.optim_batch_size
        self.num_trains_per_train_loop = self.args.num_trains_per_train_loop

        # SAC hyperparameters
        self.soft_target_tau = 5e-3
        self.target_update_period = 1
        self.reward_scale = 1.0

        if hasattr(args, 'target_entropy'):
            self.target_entropy = args.target_entropy
        else:
            self.target_entropy = -args.agent_action_dim  # heuristic value from Tuomas

        self._n_train_steps_total = 0


    def unpack_batch(self, batch):
        """
            NOTE:
                terminals = 1 - masks (verified)
                terminals = done
        """
        states = torch.from_numpy(np.stack(batch.state)).to(torch.float32).to(self.device)  # (bsize, sdim)
        actions = torch.from_numpy(np.stack(batch.action)).to(torch.float32).to(self.device)  # (bsize, adim)
        next_states = torch.from_numpy(np.stack(batch.next_state)).to(torch.float32).to(self.device)  # (bsize, sdim)
        masks = torch.from_numpy(np.stack(batch.mask)).to(torch.float32).to(self.device).unsqueeze(-1)  # (bsize, 1)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(torch.float32).to(self.device).unsqueeze(-1)  # (bsize, 1)
        terminals = (1 - masks)  # terminals = done (verified)
        assert actions.dim() == 2 and (states.dim() == 2 or states.dim() == 4)
        assert masks.dim() == rewards.dim() == 2
        assert masks.shape[-1] == rewards.shape[-1] == 1
        return states, actions, next_states, terminals, rewards

    def improve(self, agent):
        ######################################################################
        # TODO: self.num_trains_per_train_loop
        for step in range(self.num_trains_per_train_loop):
            batch = agent.replay_buffer.sample(self.optim_batch_size)  # with replacement
            states, actions, next_states, terminals, rewards = self.unpack_batch(batch)
            self.sac_step(agent, states, actions, next_states, terminals, rewards)
        ######################################################################

    def sac_step(self, agent, states, actions, next_states, terminals, rewards):
        """
            1. unpack batch (note if terminals and masks are flipped!)
            2. run_policy
            3. update alpha
            4. get policy loss
            5. run Q functions
            6. get next actions from policy
            7. get target Q values
            8. get Bellman target
            9. get Q function losses
            10. backward() step()
            11. polyak averaging
        """

        """
        Policy and Alpha Loss
        """
        ######################################################################
        # 2 SHOULD WORK (tested)
        """
            new_obs_actions: (bsize, adim)
            log_pi: (bsize, adim)

            NOTE: you should probably replace log_pi with the actual entropy
        """
        new_obs_actions, new_obs_actions_dist = agent.policy.select_action(
            states, deterministic=False, reparameterize=True)
        log_pi = agent.policy.get_log_prob(new_obs_actions_dist, new_obs_actions)  # has a gradient
        ######################################################################
        ######################################################################
        # 3 TODO: (tested)
        """
            (log_pi + self.target_entropy).detach(): (bsize, adim)
            agent.log_alpha * (log_pi + self.target_entropy).detach(): (bsize, adim)
            -(agent.log_alpha * (log_pi + self.target_entropy).detach()).mean(): ()

            # TODO: make sure broadcasting works between log_alpha and log_pi
        """
        if self.args.use_automatic_entropy_tuning:
            alpha_loss = -(agent.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            agent.optimizers['alpha'].zero_grad()
            alpha_loss.backward()
            agent.optimizers['alpha'].step()
            alpha = agent.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1
        ######################################################################
        ######################################################################
        # 4 VERBATIM (tested)
        """
            q_new_actions: (bsize, adim)
            alpha*log_pi: (bsize, adim)
            alpha*log_pi - q_new_actions: (bsize, adim)
            policy_loss: ()
        """
        q_new_actions = torch.min(
            agent.qf1(states, new_obs_actions),
            agent.qf2(states, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()
        ######################################################################

        """
        QF Loss
        """
        ######################################################################
        # 5 VERBATIM (tested)
        """
            q1_pred: (bsize, adim)
            q2_pred: (bsize, adim)
        """
        q1_pred = agent.qf1(states, actions)
        q2_pred = agent.qf2(states, actions)
        ######################################################################
        ######################################################################
        # 6 SHOULD WORK (tested)
        """
            new_next_actions: (bsize, adim)
            new_log_pi: (bsize, adim)
        """
        new_next_actions, new_next_actions_dist = agent.policy.select_action(
            next_states, deterministic=False, reparameterize=True)
        new_log_pi = agent.policy.get_log_prob(new_next_actions_dist, new_next_actions)
        ######################################################################
        ######################################################################
        # 7 VERBATIM (tested)
        """
            agent.target_qf1(next_states, new_next_actions): (bsize, adim)
            agent.target_qf2(next_states, new_next_actions): (bsize, adim)
            target_q_values: (bsize, adim)
        """
        target_q_values = torch.min(
            agent.target_qf1(next_states, new_next_actions),
            agent.target_qf2(next_states, new_next_actions),
            ) - alpha * new_log_pi
        ######################################################################

        ######################################################################
        # 8 (tested)
        """
            (1. - terminals) * self.gamma * target_q_values: (bsize, 1)
            q_target: (bsize, adim)

        """
        q_target = self.reward_scale * rewards + (1. - terminals) * self.gamma * target_q_values
        ######################################################################

        ######################################################################
        # 9 VERBATIM: (tested)
        """
            qf1_loss: ()
            qf2_loss: ()
        """
        qf1_loss = F.mse_loss(q1_pred, q_target.detach())
        qf2_loss = F.mse_loss(q2_pred, q_target.detach())
        ######################################################################

        """
        Update networks
        """
        ######################################################################
        # 10 VERBATIM: (tested)

        agent.optimizers['qf1'].zero_grad()
        agent.optimizers['qf2'].zero_grad()
        agent.optimizers['policy'].zero_grad()


        qf1_loss.backward()
        qf2_loss.backward()
        policy_loss.backward()

        agent.optimizers['qf1'].step()
        agent.optimizers['qf2'].step()
        agent.optimizers['policy'].step()
        ######################################################################

        """
        Polyak Averaging
        """
        ######################################################################
        # 11 (tested)
        """
            it's possible that you have a lower norm than both after averaging!
            this is possible if the vectors originally had a dot product of less than 0
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                agent.qf1, agent.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                agent.qf2, agent.target_qf2, self.soft_target_tau
            )
        self._n_train_steps_total += 1
        ######################################################################

