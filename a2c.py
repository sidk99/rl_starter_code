import numpy as np
import torch
import torch.nn.functional as F

eps = np.finfo(np.float32).eps.item()

class A2C():
    def __init__(self, device, args):#gamma=0.99):
        self.device = device
        self.gamma = args.gamma
        self.max_buffer_size = 100

    def improve(self, agent):
        batch = agent.buffer.sample()
        states = torch.from_numpy(np.stack(batch.state)).to(torch.float32)
        actions = torch.from_numpy(np.stack(batch.action)).to(torch.float32)
        values = agent.valuefn(states)
        log_probs = agent.policy.get_log_prob(states, actions)

        R = 0
        policy_losses = []
        value_losses = []
        rewards = []
        for r in batch.reward[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, value, r in zip(log_probs, values, rewards):
            reward = r - value.item()
            policy_losses.append(-log_prob * reward)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
        agent.policy_optimizer.zero_grad()
        agent.value_optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        agent.policy_optimizer.step()
        agent.value_optimizer.step()
        agent.buffer.clear_buffer()
