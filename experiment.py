import numpy as np
import torch

from log import RunningAverage


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    return log

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
            action, log_prob, value = self.agent(state, deterministic=deterministic)
            state, reward, done, _ = self.env.step(action)
            if self.args.render:
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
            break

        all_returns = [e['return'] for e in all_episodes_data]
        stats = {
            'avg_return': np.mean(all_returns),
            'min_return': np.min(all_returns),
            'max_return': np.max(all_returns),
            'total_return': np.sum(all_returns),
            'num_episodes': len(all_episodes_data)
        }
        # print('num_episodes', stats['num_episodes'])
        return all_episodes_data, stats

    def train(self, max_episodes):
        run_avg = RunningAverage()
        for i_episode in range(max_episodes):
            episode_data, stats = self.collect_samples(deterministic=False)
            ret = stats['avg_return'] # but we shouldn't do a running avg actually
            # ret = sum([e['reward'] for e in episode_data])
            running_return = run_avg.update_variable('reward', ret)
            if i_episode % self.args.update_every == 0:
                # print('buffer size: {}'.format(len(self.agent.buffer)))
                self.rl_alg.improve(self.agent)
            if i_episode % self.args.log_every == 0:
                # print('Episode {}\tLast Return: {:.2f}\tAverage Return: {:.2f}'.format(
                #     i_episode, ret, running_return))
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
