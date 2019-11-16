import copy
import math
import numpy as np
import pickle
import sys
import torch
import torch.nn.functional as F

from collections import defaultdict

from starter_code.common import estimate_advantages
import starter_code.utils as u

eps = np.finfo(np.float32).eps.item()

def analyze_size(obj, obj_name):
    obj_pickle = pickle.dumps(obj)
    print('Size of {}: {}'.format(obj_name, sys.getsizeof(obj_pickle)))

def rlalg_switch(alg_name):
    rlalgs = {
        'ppo': PPO,
        'a2c': A2C,
        'vpg': VPG
    }
    return rlalgs[alg_name]


class VPG():
    def __init__(self, device, args):
        self.device = device
        self.gamma = 0.99
        self.max_buffer_size = 100

    def unpack_batch(self, batch):
        states = torch.from_numpy(np.stack(batch.state)).to(torch.float32).to(self.device)  # (bsize, sdim)
        actions = torch.from_numpy(np.stack(batch.action)).to(torch.float32).to(self.device)  # (bsize, adim)
        assert actions.dim() == 2 and (states.dim() == 2 or states.dim() == 4)
        return states, actions

    def improve(self, agent):
        batch = agent.replay_replay_buffer.sample()
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
        agent.policy_optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum()
        loss.backward()
        agent.policy_optimizer.step()
        agent.replay_buffer.clear_buffer()


class A2C():
    def __init__(self, device, args):
        self.device = device
        self.gamma = 0.99
        self.max_buffer_size = 100

    def unpack_batch(self, batch):
        states = torch.from_numpy(np.stack(batch.state)).to(torch.float32).to(self.device)  # (bsize, sdim)
        actions = torch.from_numpy(np.stack(batch.action)).to(torch.float32).to(self.device)  # (bsize, adim)
        assert actions.dim() == 2 and (states.dim() == 2 or states.dim() == 4)
        return states, actions

    def improve(self, agent):
        batch = agent.replay_replay_buffer.sample()
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
        agent.policy_optimizer.zero_grad()
        agent.value_optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        agent.policy_optimizer.step()
        agent.value_optimizer.step()
        agent.replay_replay_buffer.clear_buffer()


class PPO():
    def __init__(self, device, args):
        self.device = device
        self.args = args
        self.max_buffer_size = 4096

        self.gamma = 0.99
        self.tau = 0.95 
        self.l2_reg = 1e-3
        self.clip_epsilon = 0.2
        self.entropy_coeff = args.entropy_coeff

        self.optim_epochs = 10
        self.optim_batch_size = 256
        self.optim_value_iternum = 1

        self.max_buffer_size = 100
        self.optim_batch_size = 10

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
        states = torch.from_numpy(np.stack(batch.state)).to(torch.float32).to(self.device)  # (bsize, sdim)
        actions = torch.from_numpy(np.stack(batch.action)).to(torch.float32).to(self.device)  # (bsize, adim)
        masks = torch.from_numpy(np.stack(batch.mask)).to(torch.float32).to(self.device)  # (bsize)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(torch.float32).to(self.device)  # (bsize)
        assert actions.dim() == 2 and (states.dim() == 2 or states.dim() == 4)
        assert masks.dim() == rewards.dim() == 1
        return states, actions, masks, rewards

    def improve(self, agent):

        # if agent.id == 0:
        #     print('parameters before update for agent 0 (inside)')
        #     u.visualize_parameters(agent, print)


        self.reset_record()
        batch = agent.replay_buffer.sample()
        states, actions, masks, rewards = self.unpack_batch(batch)
        with torch.no_grad():
            values = agent.valuefn(states)  # (bsize, 1)
            fixed_log_probs = agent.policy.get_log_prob(states, actions)  # (bsize, 1)
            assert values.dim() == fixed_log_probs.dim() == 2

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau, self.device)  # (bsize, 1) (bsize, 1)

        """perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(states.shape[0] / self.optim_batch_size))
        for j in range(self.optim_epochs):
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

        # analyze_size(agent.replay_buffer, 'before inside clearing: agent {} replay_buffer'.format(agent.id))
        # agent.replay_buffer.clear_buffer()
        # analyze_size(agent.replay_buffer, 'after inside clearing: agent {} replay_buffer'.format(agent.id))

        # if agent.id == 0:
        #     print('parameters after update for agent 0 (inside)')
        #     u.visualize_parameters(agent, print)

        # print('Agent {} just cleared buffer in improve'.format(agent.id))

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
            agent.value_optimizer.zero_grad()
            value_loss.backward()
            agent.value_optimizer.step()

        """update policy"""
        log_probs = agent.policy.get_log_prob(states, actions)
        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_surr = -torch.min(surr1, surr2).mean()
        entropy = agent.policy.get_entropy(states).mean()
        policy_loss = policy_surr - self.entropy_coeff*entropy
        agent.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 40)
        agent.policy_optimizer.step()



        """log"""
        num_clipped = (surr1-surr2).nonzero().size(0)
        ratio_clipped = num_clipped / states.size(0)
        # print('ratio_clipped', ratio_clipped)
        log = {}
        log['num_clipped'] = num_clipped
        log['ratio_clipped'] = ratio_clipped
        log['entropy'] = entropy
        log['bsize'] = states.size(0)
        log['value_loss'] = value_loss.item()
        log['policy_surr'] = policy_surr.item()
        log['policy_loss'] = policy_loss.item()
        return log


# class SAC():
#     def __init__(self, device, args):
#         self.device = device
#         self.args = args

        
        




#         """
#         self.max_buffer_size = 4096

#         self.gamma = 0.99
#         self.tau = 0.95 
#         self.l2_reg = 1e-3
#         self.clip_epsilon = 0.2
#         self.entropy_coeff = args.entropy_coeff

#         self.optim_epochs = 10
#         self.optim_batch_size = 256
#         self.optim_value_iternum = 1

#         self.max_buffer_size = 100
#         self.optim_batch_size = 10
#         """


#     def unpack_batch(self, batch):
#         states = torch.from_numpy(np.stack(batch.state)).to(torch.float32).to(self.device)  # (bsize, sdim)
#         actions = torch.from_numpy(np.stack(batch.action)).to(torch.float32).to(self.device)  # (bsize, adim)
#         masks = torch.from_numpy(np.stack(batch.mask)).to(torch.float32).to(self.device)  # (bsize)
#         rewards = torch.from_numpy(np.stack(batch.reward)).to(torch.float32).to(self.device)  # (bsize)
#         assert actions.dim() == 2 and (states.dim() == 2 or states.dim() == 4)
#         assert masks.dim() == rewards.dim() == 1
#         return states, actions, masks, rewards

#     def improve(self, agent):
#         self.reset_record()
#         batch = agent.replay_buffer.sample()
#         states, actions, masks, rewards = self.unpack_batch(batch)
#         with torch.no_grad():
#             values = agent.valuefn(states)  # (bsize, 1)
#             fixed_log_probs = agent.policy.get_log_prob(states, actions)  # (bsize, 1)
#             assert values.dim() == fixed_log_probs.dim() == 2

#         """get advantage estimation from the trajectories"""
#         advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau, self.device)  # (bsize, 1) (bsize, 1)

#         """perform mini-batch PPO update"""
#         optim_iter_num = int(math.ceil(states.shape[0] / self.optim_batch_size))
#         for j in range(self.optim_epochs):
#             perm = np.arange(states.shape[0])
#             np.random.shuffle(perm)
#             perm = torch.LongTensor(perm).to(self.device)

#             states, actions, returns, advantages, fixed_log_probs = \
#                 states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

#             for i in range(optim_iter_num):
#                 ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, states.shape[0]))
#                 states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
#                     states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

#                 minibatch_log = self.ppo_step(agent, states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b)
#                 self.record(minibatch_log=minibatch_log, epoch=j, iter=i)
#         agent.replay_buffer.clear_buffer()


