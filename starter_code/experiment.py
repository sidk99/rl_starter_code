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
        self.run_avg = RunningAverage()

    def get_bids_for_episode(self, episode_info):
        episode_bids = defaultdict(lambda: [])
        for step in episode_info['organism_episode_data']:
            probs = list(step['action_dist'].probs.detach()[0].cpu().numpy())
            for index, prob in enumerate(probs):
                episode_bids[index].append(prob)
        return episode_bids

    def sample_episode(self, env, deterministic, render):
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

        episode_info = {
            'organism_episode_data': episode_data,
            'episode_stats': stats
        }
        return episode_info

    def collect_samples(self, deterministic):
        num_steps = 0
        num_episodes = 0
        all_returns = []
        all_moves = []

        while num_steps < self.rl_alg.max_buffer_size:
            train_env_manager = self.task_progression.sample(i=self.epoch, mode='train')
            episode_info = self.sample_episode(env=train_env_manager.env, deterministic=deterministic, render=False)
            all_returns.append(episode_info['episode_stats']['return'])
            all_moves.append(episode_info['episode_stats']['steps'])
            num_steps += (episode_info['episode_stats']['steps'])
            num_episodes += 1
        stats = {
            'mean_return': np.mean(all_returns),
            'min_return': np.min(all_returns),
            'max_return': np.max(all_returns),
            'std_return': np.std(all_returns),
            'total_return': np.sum(all_returns),
            'mean_moves': np.mean(all_moves),
            'std_moves': np.std(all_moves),
            'min_moves': np.min(all_moves),
            'max_moves': np.max(all_moves),
            'num_episodes': num_episodes
        }
        self.run_avg.update_variable('reward', stats['mean_return'])

        episode_info['epoch_stats'] = stats
        return episode_info


    def train(self, max_epochs):
        for epoch in range(max_epochs):
            if epoch % self.args.eval_every == 0:
                stats = self.eval(epoch=epoch)  # this needs to be done for multiple environments.

                for mode in ['train', 'test']:
                    for env_manager in self.task_progression[self.epoch][mode]:
                        env_manager.saver.save(epoch, 
                            {'args': self.args,
                             'epoch': epoch,
                             'logger': self.logger.get_state_dict(),
                             'experiment': stats[env_manager.env_name],
                             'organism': self.organism.get_state_dict()},
                             self.logger.printf)

            epoch_info = self.collect_samples(deterministic=False)


            metrics = ['min_return', 'max_return', 'mean_return', 'std_return',
                   'min_moves', 'max_moves', 'mean_moves', 'std_moves']

            self.logger.update_variable(name='epoch', index=epoch, value=epoch)
            for metric in metrics:
                self.logger.update_variable(
                    name=metric, index=epoch, value=epoch_info['epoch_stats'][metric], include_running_avg=True)
            self.logger.plot(
                var_pairs=[(('epoch', k)) for k in metrics],
                expname=self.logger.expname,
                pfunc=self.logger.printf)

            self.logger.saver.save(epoch, 
                {'args': self.args,
                 'epoch': epoch,
                 'logger': self.logger.get_state_dict(),
                 'experiment': epoch_info['epoch_stats'],
                 'organism': self.organism.get_state_dict()},
                 self.logger.printf)


            if epoch >= self.args.anneal_policy_lr_after:
                self.organism.step_optimizer_schedulers(self.logger.printf)

            self.organism.update(self.rl_alg)

            if epoch % self.args.log_every == 0:
                self.log(epoch, epoch_info)

        self.finish_training()

    def finish_training(self):
        # env.close()
        # plt.close()
        pass

    def log(self, epoch, epoch_info):
        stats = epoch_info['epoch_stats']
        self.logger.printf('Episode {}\tAvg Return: {:.2f}\tMin Return: {:.2f}\tMax Return: {:.2f}\tRunning Return: {:.2f}'.format(
            epoch, stats['mean_return'], stats['min_return'], stats['max_return'], self.run_avg.get_value('reward')))

    def test(self, epoch, env_manager, num_test, visualize):
        returns = []
        moves = []
        for i in tqdm(range(num_test)):
            with torch.no_grad():
                episode_info = self.sample_episode(env=env_manager.env, deterministic=False, render=visualize)
                ret = episode_info['episode_stats']['return']
                if i == 0 and visualize and self.organism.discrete:
                    bids = self.get_bids_for_episode(episode_info)
                    env_manager.save_video(epoch, i, bids, ret, episode_info['organism_episode_data'])
            returns.append(ret)
            moves.append(episode_info['episode_stats']['steps'])
        returns = np.array(returns)
        moves = np.array(moves)
        stats = {'returns': returns,
                 'mean_return': np.mean(returns),
                 'std_return': np.std(returns),
                 'min_return': np.min(returns),
                 'max_return': np.max(returns),
                 'moves': moves,
                 'mean_moves': np.mean(moves),
                 'std_moves': np.std(moves),
                 'min_moves': np.min(moves),
                 'max_moves': np.max(moves),}
        return stats

    def eval(self, epoch):
        metrics = ['min_return', 'max_return', 'mean_return', 'std_return',
                   'min_moves', 'max_moves', 'mean_moves', 'std_moves']
        multi_task_stats = {} 
        for mode in ['train', 'test']:
            for env_manager in self.task_progression[self.epoch][mode]:
                stats = self.test(epoch, env_manager, num_test=10, visualize=True)

                env_manager.update_variable(name='epoch', index=epoch, value=epoch)
                for metric in metrics:
                    env_manager.update_variable(
                        name=metric, index=epoch, value=stats[metric], include_running_avg=True)
                env_manager.plot(
                    var_pairs=[(('epoch', k)) for k in metrics],
                    expname=env_manager.env_name,
                    pfunc=self.logger.printf)

                self.logger.pprintf(stats)
                multi_task_stats[env_manager.env_name] = stats
        return multi_task_stats  # need to fix this!
        # return stats  # for now


