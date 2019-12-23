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
from starter_code.sampler import Sampler, AgentStepInfo, Centralized_RL_Stats, collect_train_samples_serial, collect_train_samples_parallel
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
            s += delim + '{}: {:.5f}'.format(k, v)
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
        self.run_avg = RunningAverage()
        self.run_avg.update_variable('steps', 0)
        self.metrics = ['min_return', 'max_return', 'mean_return', 'std_return',
                       'min_steps', 'max_steps', 'mean_steps', 'std_steps']

        self.parallel_collect = True

    def collect_samples(self, epoch):
        """
            Here we will be guaranteed to collect exactly self.rl_alg.num_samples_before_update samples
        """
        t0 = time.time()

        gt.stamp('Epoch {}: Before Collect Samples'.format(epoch))
        collector = collect_train_samples_parallel if self.parallel_collect else collect_train_samples_serial

        stats_collector = collector(
            epoch=epoch, 
            max_steps=self.rl_alg.num_samples_before_update, 
            objects=AttrDict(
                task_progression=self.task_progression, 
                stats_collector_builder=self.stats_collector_builder, 
                sampler_builder=self.exploration_sampler_builder, 
                organism=self.organism)
            )

        stats = stats_collector.bundle_batch_stats()  # you might want to bundle batch stats after the parallel thing

        for episode_data in stats_collector.data['episode_datas']:
            for e in episode_data:
                self.organism.store_transition(e)

        self.run_avg.update_variable('mean_return', stats['mean_return'])
        self.run_avg.update_variable('steps', self.run_avg.get_last_value('steps')+stats['total_steps'])

        gt.stamp('Epoch {}: After Collect Samples'.format(epoch))

        self.logger.printf('Epoch {}: Time to Collect Samples: {}'.format(epoch, time.time()-t0))
        return stats


    def main_loop(self, max_epochs):
        # populate replay buffer before training
        for epoch in gt.timed_for(range(max_epochs)):
            if epoch % self.args.eval_every == 0:
                self.eval_step(epoch)
            self.train_step(epoch)
        self.finish_training()


    def train_step(self, epoch):
        epoch_stats = self.collect_samples(epoch)
        if epoch % self.args.log_every == 0:
            self.log(epoch, epoch_stats)
        if epoch % self.args.visualize_every == 0:
            # this ticks the epoch and steps
            self.visualize(self.logger, epoch, epoch_stats, self.logger.expname)
        if epoch % self.args.save_every == 0:
            self.save(self.logger, epoch, epoch_stats)
        self.update(epoch)


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
        self.logger.printf('Epoch {}: Time to Update: {}'.format(epoch, time.time()-t0))
        gt.stamp('Epoch {}: After Update'.format(epoch))
        self.organism.clear_buffer()
        torch.cuda.empty_cache()  # may be I should do this

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
                evaluation_sampler = self.evaluation_sampler_builder(self.organism)
                episode_data = evaluation_sampler.sample_episode(
                    env=env_manager.env, 
                    # organism=self.organism,
                    max_timesteps_this_episode=env_manager.max_episode_length)
                ##########################################
                if i == 0 and self.organism.discrete and env_manager.visual:

                    bids = evaluation_sampler.get_bids_for_episode(episode_data)
                    returns = sum([e.reward for e in episode_data])
                    frames = [e.frame for e in episode_data]

                    env_manager.save_video(epoch, i, bids, returns, frames)

                ##########################################
            stats_collector.append(episode_data, eval_mode=True)
        stats = stats_collector.bundle_batch_stats(eval_mode=True)
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

    def visualize(self, env_manager, epoch, stats, name, eval_mode=False):
        self.update_metrics(env_manager, epoch, stats)
        self.plot_metrics(env_manager, name)

    def eval_step(self, epoch):
        for env_manager in self.task_progression[epoch]['test']:
            stats = self.test(epoch, env_manager, num_test=self.args.num_test)
            self.logger.pprintf({k:v for k,v in stats.items() if k not in 
                ['bid_differences', 'Q_differences', 'agent_episode_data', 'organism_episode_data']})
            if epoch % self.args.visualize_every == 0:
                self.visualize(env_manager, epoch, stats, env_manager.env_name)
            if epoch % self.args.save_every == 0:
                self.save(env_manager, epoch, stats)


class CentralizedExperiment(Experiment):
    def __init__(self, agent, task_progression, rl_alg, logger, device, args):
        super(CentralizedExperiment, self).__init__(agent, task_progression, rl_alg, logger, device, args)
        self.exploration_sampler_builder = lambda organism: Sampler(
            organism=organism,
            eval_mode=False, 
            step_info=AgentStepInfo, 
            deterministic=False, 
            render=False, 
            device=device)
        self.evaluation_sampler_builder = lambda organism: Sampler(
            organism=organism,
            eval_mode=True, 
            step_info=AgentStepInfo, 
            deterministic=False, 
            render=True, 
            device=device)

        self.stats_collector_builder = Centralized_RL_Stats

        # wait yeah, actually you only need the filter for the exploration sampler!




