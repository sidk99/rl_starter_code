from collections import defaultdict, OrderedDict
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

from starter_code.log import RunningAverage
from starter_code.sampler import Sampler, AgentStepInfo, Centralized_RL_Stats
import starter_code.utils as u
from starter_code.utils import AttrDict, is_float



"""
* Algorithm level (epoch, episode, learning rate, etc)
* Monitoring level (wall clock time, etc)
* MDP level (returns, etc)
"""

def analyze_size(obj, obj_name):
    obj_pickle = pickle.dumps(obj)
    print('Size of {}: {}'.format(obj_name, sys.getsizeof(obj_pickle)))

def log_string(ordered_dict):
    s = ''
    for i, (k, v) in enumerate(ordered_dict.items()):
        delim = '' if i == 0 else ' | '
        if is_float(v):
            s += delim + '{}: {:.2f}'.format(k, v)
        else:
            s += delim + '{}: {}'.format(k, v)
    return s


class Experiment():
    """
        args.eval_every
        args.anneal_policy_lr_after
        args.log_every
        args.num_test
    """
    def __init__(self, agent, task_progression, rl_alg, logger, device, args):
        self.organism = agent
        self.task_progression = task_progression
        self.rl_alg = rl_alg
        self.logger = logger
        self.device = device
        self.args = args
        self.epoch = 0
        self.run_avg = RunningAverage()
        self.run_avg.update_variable('steps', 0)
        self.metrics = ['min_return', 'max_return', 'mean_return', 'std_return',
                       'min_steps', 'max_steps', 'mean_steps', 'std_steps']

    def collect_samples(self, epoch):
        """
            Here we will be guaranteed to collect exactly self.rl_alg.num_samples_before_update samples
        """
        gt.stamp('Epoch {}: Before Collect Samples'.format(epoch))
        num_steps = 0
        stats_collector = self.stats_collector_builder()
        # reset the data structure here? At least this is how things have been going anyways
        # yeah this would just contain data from the current batch

        while num_steps < self.rl_alg.num_samples_before_update:
            train_env_manager = self.task_progression.sample(i=self.epoch, mode='train')

            max_timesteps_this_episode = min(
                self.rl_alg.num_samples_before_update - num_steps,
                train_env_manager.max_episode_length)

            ################################################################
            episode_info = self.exploration_sampler.sample_episode(
                env=train_env_manager.env, 
                organism=self.organism,
                max_timesteps_this_episode=max_timesteps_this_episode)

            ################################################################
            stats_collector.append(episode_info)
            num_steps += (episode_info.steps)

        stats = stats_collector.bundle_batch_stats()

        self.run_avg.update_variable('mean_return', stats['mean_return'])
        self.run_avg.update_variable('steps', self.run_avg.get_last_value('steps')+num_steps)

        gt.stamp('Epoch {}: After Collect Samples'.format(epoch))
        return stats

    def train(self, max_epochs):
        # populate replay buffer before training
        for epoch in gt.timed_for(range(max_epochs)):
            # print('epoch: {}'.format(epoch))
            if epoch % self.args.eval_every == 0:
                self.eval(epoch=epoch)

            epoch_stats = self.collect_samples(epoch)
            
            if epoch % self.args.log_every == 0:
                self.log(epoch, epoch_stats)

            if epoch % self.args.visualize_every == 0:
                # this ticks the epoch and steps
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
        self.logger.printf(log_string(OrderedDict({
            'epoch': epoch,
            'env steps this batch': epoch_stats['total_steps'],
            'env steps taken': self.run_avg.get_last_value('steps'),
            'avg return': epoch_stats['mean_return'],
            'min return': epoch_stats['min_return'],
            'max return': epoch_stats['max_return'],
            'running mean return': self.run_avg.get_value('mean_return')
            })))

    def test(self, epoch, env_manager, num_test):
        stats_collector = self.stats_collector_builder()
        for i in tqdm(range(num_test)):
            with torch.no_grad():
                episode_info = self.evaluation_sampler.sample_episode(
                    env=env_manager.env, 
                    organism=self.organism,
                    max_timesteps_this_episode=env_manager.max_episode_length)
                ##########################################
                if i == 0 and self.organism.discrete:
                    env_manager.save_video(epoch, i, episode_info.bids, episode_info.returns, episode_info.frames)
                ##########################################
            stats_collector.append(episode_info)
        stats = stats_collector.bundle_batch_stats()
        return stats

    def update_metrics(self, env_manager, epoch, stats):
        env_manager.update_variable(name='epoch', index=epoch, value=epoch)
        # this assumes that you save every whole number of epochs
        env_manager.update_variable(name='steps', index=epoch, value=self.run_avg.get_last_value('steps'))

        for metric in self.metrics:
            env_manager.update_variable(
                name=metric, index=epoch, value=stats[metric], include_running_avg=True)

    def plot_metrics(self, env_manager, name):
        env_manager.plot(
            var_pairs=[(('steps', k)) for k in self.metrics],
            expname=name,
            pfunc=self.logger.printf)

    def save(self, env_manager, epoch, stats):
        # this should also be based on the running reward.
        env_manager.saver.save(epoch, 
            {'args': self.args,
             'epoch': epoch,
             'logger': self.logger.get_state_dict(),
             'mean_return': stats['mean_return'],  # note I'm not saving the entire stats dict anymore!
             'organism': self.organism.get_state_dict()},
             self.logger.printf)

    def visualize(self, env_manager, epoch, stats, name):
        self.update_metrics(env_manager, epoch, stats)
        self.plot_metrics(env_manager, name)

    def eval(self, epoch):
        for env_manager in self.task_progression[self.epoch]['test']:
            stats = self.test(epoch, env_manager, num_test=self.args.num_test)
            # self.logger.pprintf(stats)
            self.logger.pprintf({k:v for k,v in stats.items() if k not in ['agent_episode_data', 'organism_episode_data']})
            # hold on - why are we saving this?
            if epoch % self.args.visualize_every == 0:
                self.visualize(env_manager, epoch, stats, env_manager.env_name)
            if epoch % self.args.save_every == 0:
                self.save(env_manager, epoch, stats)


class CentralizedExperiment(Experiment):
    def __init__(self, agent, task_progression, rl_alg, logger, device, args):
        super(CentralizedExperiment, self).__init__(agent, task_progression, rl_alg, logger, device, args)
        self.exploration_sampler = Sampler(
            step_info=AgentStepInfo, deterministic=False, render=False, device=device)
        self.evaluation_sampler = Sampler(
            step_info=AgentStepInfo, deterministic=False, render=True, device=device)
        self.stats_collector_builder = Centralized_RL_Stats
