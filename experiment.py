import numpy as np
import torch

from log import RunningAverage

import ipdb

class Experiment():
    def __init__(self, agent, env, rl_alg, args):
        self.agent = agent
        self.env = env
        self.rl_alg = rl_alg
        self.args = args

    def sample_trajectory(self, deterministic):
        episode_data = []
        state = self.env.reset()
        for t in range(self.rl_alg.max_buffer_size):  # Don't infinite loop while learning
            state_var = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                action = self.agent(state_var, deterministic=deterministic)
            action = action[0]
            next_state, reward, done, _ = self.env.step(action)
            if self.args.render:
                self.env.render()
            mask = 0 if done else 1
            e = {
                 'state': state,
                 'action': action,
                 'mask': mask,
                 'reward': reward,
                 }
            episode_data.append(e)
            self.agent.store_transition(e)
            if done:
                break
            state = next_state
        stats = {'return': sum([e['reward'] for e in episode_data]),
                 'steps': t}
        return episode_data, stats

    def collect_samples(self, deterministic):
        num_steps = 0
        all_episodes_data = []

        while num_steps < self.rl_alg.max_buffer_size:
            episode_data, episode_stats= self.sample_trajectory(deterministic)
            all_episodes_data.append(episode_stats)
            num_steps += (episode_stats['steps']+1)

        all_returns = [e['return'] for e in all_episodes_data]
        stats = {
            'avg_return': np.mean(all_returns),
            'min_return': np.min(all_returns),
            'max_return': np.max(all_returns),
            'total_return': np.sum(all_returns),
            'num_episodes': len(all_episodes_data)
        }
        return all_episodes_data, stats

    def train(self, max_episodes):
        run_avg = RunningAverage()
        for i_episode in range(max_episodes):
            episode_data, stats = self.collect_samples(deterministic=False)
            ret = stats['avg_return'] # but we shouldn't do a running avg actually
            running_return = run_avg.update_variable('reward', ret)
            if i_episode % self.args.update_every == 0:
                self.rl_alg.improve(self.agent)
            if i_episode % self.args.log_every == 0:
                print('Episode {}\tAvg Return: {:.2f}\tMin Return: {:.2f}\tMax Return: {:.2f}\tRunning Return: {:.2f}'.format(
                    i_episode, stats['avg_return'], stats['min_return'], stats['max_return'], running_return))

            if i_episode % self.args.eval_every == 0:
                stats = self.test(max_episodes=10)
                print(stats)

    def test(self, max_episodes):
        returns = []
        for i_episode in range(max_episodes):
            with torch.no_grad():
                episode_data, stats = self.sample_trajectory(deterministic=True)
                ret = stats['return']
            returns.append(ret)
        returns = np.array(returns)
        stats = {'mean return': np.mean(returns),
                 'std return': np.std(returns),
                 'min return': np.min(returns),
                 'max return': np.max(returns)}
        return stats
