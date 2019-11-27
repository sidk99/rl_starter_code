from collections import defaultdict
import gtimer as gt
import ipdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import time
import torch
from tqdm import tqdm

from log import RunningAverage
from starter_code.sampler import Sampler
import starter_code.utils as u
from starter_code.utils import AttrDict

def analyze_size(obj, obj_name):
    obj_pickle = pickle.dumps(obj)
    print('Size of {}: {}'.format(obj_name, sys.getsizeof(obj_pickle)))

class Experiment():
    """
        args.eval_every
        args.anneal_policy_lr_after
        args.log_every
        args.num_test
    """
    def __init__(self, agent, task_progression, rl_alg, exploration_sampler, evaluation_sampler, logger, device, args):
        self.organism = agent
        self.task_progression = task_progression
        self.rl_alg = rl_alg
        self.logger = logger
        self.device = device
        self.args = args
        self.epoch = 0
        self.run_avg = RunningAverage()
        self.metrics = ['min_return', 'max_return', 'mean_return', 'std_return',
                       'min_moves', 'max_moves', 'mean_moves', 'std_moves']
        self.exploration_sampler = exploration_sampler
        self.evaluation_sampler = evaluation_sampler


    # def foo(self):
    #     my_array = []
    #     for i in range(loop):
    #         x = bar()
    #         my_array.append(x)
    #     stats = compute_stats(my_array)
    #     return stats


    def collect_samples(self, epoch):
        """
            Here we will be guaranteed to collect exactly self.rl_alg.num_samples_before_update samples
        """
        gt.stamp('Epoch {}: Before Collect Samples'.format(epoch))
        num_steps = 0
        num_episodes = 0
        all_returns = []
        all_moves = []

        while num_steps < self.rl_alg.num_samples_before_update:
            train_env_manager = self.task_progression.sample(i=self.epoch, mode='train')

            max_timesteps_this_episode = min(
                self.rl_alg.num_samples_before_update - num_steps,
                train_env_manager.max_episode_length)

            print('num_steps: {} num_episodes: {}, max_timesteps_this_episode: {}, max_episode_length_from_env: {}'.format(num_steps, num_episodes, max_timesteps_this_episode, train_env_manager.max_episode_length))

            ################################################################
            episode_info = self.exploration_sampler.sample_episode(
                env=train_env_manager.env, 
                organism=self.organism,
                max_timesteps_this_episode=max_timesteps_this_episode)
            ################################################################
            all_returns.append(episode_info.returns)
            all_moves.append(episode_info.moves)
            num_steps += (episode_info.moves)

            num_episodes += 1
        assert np.sum(all_moves) == num_steps
        print('num_steps collected: {}'.format(np.sum(all_moves)))

        stats = self.bundle_batch_stats(num_episodes, all_returns, all_moves)

        self.run_avg.update_variable('reward', stats['mean_return'])

        print('num_steps: {} num_episodes: {}'.format(np.sum(all_moves), num_episodes))
        gt.stamp('Epoch {}: After Collect Samples'.format(epoch))
        return stats

    def train(self, max_epochs):
        # populate replay buffer before training
        for epoch in gt.timed_for(range(max_epochs)):
            print('epoch: {}'.format(epoch))
            if epoch % self.args.eval_every == 0:
                stats = self.eval(epoch=epoch)

            epoch_stats = self.collect_samples(epoch)
            
            if epoch % self.args.log_every == 0:
                self.log(epoch, epoch_stats)

            if epoch % self.args.visualize_every == 0:
                self.visualize(self.logger, epoch, epoch_stats, self.logger.expname)
            
            if epoch % self.args.save_every == 0:
                self.save(self.logger, epoch, epoch_stats)

            self.update(epoch)

        self.finish_training()


    def finish_training(self):
        # env.close()
        # plt.close()
        # print(gt.report())
        self.logger.printf(self.organism.get_summary())
        self.clean()

    def clean(self):
        plt.close()
        if self.args.debug:
            self.logger.remove_logdir()


    def update(self, epoch):
        if epoch >= self.args.anneal_policy_lr_after:
            self.organism.step_optimizer_schedulers(self.logger.printf)

        gt.stamp('Epoch {}: Before Update'.format(epoch))
        t0 = time.time()
        self.organism.update(self.rl_alg)
        self.logger.printf('Epoch {}: After Update: {}'.format(epoch, time.time()-t0))
        gt.stamp('Epoch {}: After Update'.format(epoch))
        self.organism.clear_buffer()

    def log(self, epoch, epoch_stats):
        self.logger.printf('Epoch {}\tAvg Return: {:.2f}\tMin Return: {:.2f}\tMax Return: {:.2f}\tRunning Return: {:.2f}'.format(
            epoch, epoch_stats['mean_return'], epoch_stats['min_return'], epoch_stats['max_return'], self.run_avg.get_value('reward')))

    def test(self, epoch, env_manager, num_test):
        returns = []
        moves = []
        for i in tqdm(range(num_test)):
            with torch.no_grad():

                ################################################################
                episode_info = self.evaluation_sampler.sample_episode(
                    env=env_manager.env, 
                    organism=self.organism,
                    max_timesteps_this_episode=env_manager.max_episode_length)
                ################################################################

                if i == 0 and self.organism.discrete:
                    env_manager.save_video(epoch, i, episode_info.bids, episode_info.returns, episode_info.frames)

            returns.append(episode_info.returns)
            moves.append(episode_info.moves)

        stats = self.bundle_batch_stats(num_test, returns, moves)
        return stats

    def bundle_batch_stats(self, num_episodes, returns, moves):
        stats = dict(num_episodes=num_episodes)
        stats = dict({**stats, **self.log_metrics(np.array(returns), 'return')})
        stats = dict({**stats, **self.log_metrics(np.array(moves), 'moves')})
        return stats

    def log_metrics(self, data, label):
        """
            input
                data would be a numpy array
                label would be the name for the data
            output
                {'label': data, 
                 'mean_label': , 'std_label': , 'min_labe': , 'max_label'}
        """
        labeler = lambda cmp: '{}_{}'.format(cmp, label)
        stats = {}
        stats[label] = data
        stats[labeler('mean')] = np.mean(data)
        stats[labeler('std')] = np.std(data)
        stats[labeler('min')] = np.min(data)
        stats[labeler('max')] = np.max(data)
        stats[labeler('total')] = np.sum(data)
        return stats

    def update_metrics(self, env_manager, epoch, stats):
        for metric in self.metrics:
            env_manager.update_variable(
                name=metric, index=epoch, value=stats[metric], include_running_avg=True)

    def plot_metrics(self, env_manager, name):
        env_manager.plot(
            var_pairs=[(('epoch', k)) for k in self.metrics],
            expname=name,
            pfunc=self.logger.printf)

    def save(self, env_manager, epoch, stats):
        # this should also be based on the running reward.
        env_manager.saver.save(epoch, 
            {'args': self.args,
             'epoch': epoch,
             'logger': self.logger.get_state_dict(),
             'experiment': stats,
             'organism': self.organism.get_state_dict()},
             self.logger.printf)

    def visualize(self, env_manager, epoch, stats, name):
        env_manager.update_variable(name='epoch', index=epoch, value=epoch)
        self.update_metrics(env_manager, epoch, stats)
        self.plot_metrics(env_manager, name)

    def eval(self, epoch):
        multi_task_stats = {} 
        for env_manager in self.task_progression[self.epoch]['test']:
            stats = self.test(epoch, env_manager, num_test=self.args.num_test)
            self.logger.pprintf(stats)
            if epoch % self.args.visualize_every == 0:
                self.visualize(env_manager, epoch, stats, env_manager.env_name)
            if epoch % self.args.save_every == 0:
                self.save(env_manager, epoch, stats)
        return multi_task_stats



