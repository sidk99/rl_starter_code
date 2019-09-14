import cv2
from collections import defaultdict
import numpy as np
import torch

from log import RunningAverage

import ipdb
from tqdm import tqdm

import starter_code.env_utils as eu

class Experiment():
    def __init__(self, agent, task_progression, rl_alg, logger, device, args):
        self.organism = agent
        self.task_progression = task_progression
        self.rl_alg = rl_alg
        self.logger = logger
        self.device = device
        self.args = args
        self.epoch = 0

    def get_bids_for_episode(self, episode_data):
        episode_bids = defaultdict(lambda: [])
        for step in episode_data:
            probs = list(step['action_dist'].probs.detach()[0].cpu().numpy())
            for index, prob in enumerate(probs):
                episode_bids[index].append(prob)
        return episode_bids

    def sample_trajectory(self, env, deterministic, render):
        episode_data = []
        state = env.reset()
        #################################################
        # Debugging MiniGrid
        if type(state) == dict:
            state = state['image']
        #################################################
        for t in range(self.rl_alg.max_buffer_size):  # Don't infinite loop while learning
            state_var = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_dict = self.organism(state_var, deterministic=deterministic)
            next_state, reward, done, _ = env.step(action_dict['action'])
            #################################################
            # Debugging MiniGrid
            if type(next_state) == dict:
                next_state = next_state['image']
            #################################################
            mask = 0 if done else 1
            e = {
                 'state': state,
                 'action': action_dict['stored_action'],
                 'action_dist': action_dict['action_dist'],
                 'mask': mask,
                 'reward': reward,
                 }
            if render:
                frame = eu.render(env=env, scale=0.25)
                e['frame'] = frame
            episode_data.append(e)
            self.organism.store_transition(e)
            if done:
                break
            state = next_state
        stats = {'return': sum([e['reward'] for e in episode_data]),
                 'steps': t+1,
                 'actions': [e['action'] for e in episode_data]}
        return episode_data, stats

    def collect_samples(self, deterministic):
        num_steps = 0
        all_episodes_data = []

        while num_steps < self.rl_alg.max_buffer_size:
            train_env_manager = self.task_progression.sample(i=self.epoch, mode='train')
            episode_data, episode_stats= self.sample_trajectory(env=train_env_manager.env, deterministic=deterministic, render=False)
            all_episodes_data.append(episode_stats)
            num_steps += (episode_stats['steps'])

        all_returns = [e['return'] for e in all_episodes_data]
        stats = {
            'avg_return': np.mean(all_returns),
            'min_return': np.min(all_returns),
            'max_return': np.max(all_returns),
            'total_return': np.sum(all_returns),
            'num_episodes': len(all_episodes_data)
        }
        return all_episodes_data, stats

    def train(self, max_epochs):
        run_avg = RunningAverage()
        for epoch in range(max_epochs):
            if epoch % self.args.eval_every == 0:
                stats = self.eval(epoch=epoch)
                self.logger.saver.save(epoch, 
                    {'args': self.args,
                     'logger': self.logger.get_state_dict(),
                     'experiment': stats,
                     'agent': self.organism.get_state_dict()})

            episode_data, stats = self.collect_samples(deterministic=False)
            running_return = run_avg.update_variable('reward', stats['avg_return'])

            if epoch >= self.args.anneal_policy_lr_after:
                self.organism.step_optimizer_schedulers(self.logger.printf)

            if epoch % self.args.update_every == 0:
                self.rl_alg.improve(self.organism)

            if epoch % self.args.log_every == 0:
                self.logger.printf('Episode {}\tAvg Return: {:.2f}\tMin Return: {:.2f}\tMax Return: {:.2f}\tRunning Return: {:.2f}'.format(
                    epoch, stats['avg_return'], stats['min_return'], stats['max_return'], running_return))

    def test(self, epoch, env_manager, num_test, visualize):
        returns = []
        for i in tqdm(range(num_test)):
            with torch.no_grad():
                episode_data, stats = self.sample_trajectory(env=env_manager.env, deterministic=False, render=visualize)
                ret = stats['return']

                if i == 0 and visualize and self.organism.policy.discrete:
                    bids = self.get_bids_for_episode(episode_data)
                    env_manager.save_video(epoch, i, bids, ret, episode_data)
            returns.append(ret)
        returns = np.array(returns)
        stats = {'returns': returns,
                 'mean_return': np.mean(returns),
                 'std_return': np.std(returns),
                 'min_return': np.min(returns),
                 'max_return': np.max(returns)}
        return stats

    def eval(self, epoch):
        for mode in ['train']:
            for env_manager in self.task_progression[self.epoch][mode]:  # epoch=0 is hardcoded!!
                stats = self.test(epoch, env_manager, num_test=10, visualize=True)
                env_manager.update_variable(name='epoch', index=epoch, value=epoch)
                for metric in ['min_return', 'max_return', 'mean_return', 'std_return']:
                    env_manager.update_variable(
                        name=metric, index=epoch, value=stats[metric], include_running_avg=True)
                env_manager.plot(
                    var_pairs=[(('epoch', k)) for k in ['min_return', 'max_return', 'mean_return', 'std_return']],
                    expname=self.logger.expname)
                self.logger.pprintf(stats)
                return stats


