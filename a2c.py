import numpy as np
import torch
import torch.nn.functional as F

eps = np.finfo(np.float32).eps.item()

class A2C():
    def __init__(self, gamma=0.99):
        self.gamma = gamma

    def improve(self, agent):
        batch = agent.buffer.sample()
        R = 0
        saved_actions = zip(batch.logprob, batch.value)
        policy_losses = []
        value_losses = []
        rewards = []
        for r in batch.reward[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for (log_prob, value), r in zip(saved_actions, rewards):
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