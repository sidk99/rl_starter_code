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

from starter_code.log import ExponentialRunningAverage
from starter_code.sampler import Sampler, AgentStepInfo, Centralized_RL_Stats, collect_train_samples_serial, collect_train_samples_parallel
import starter_code.utils as u
from starter_code.utils import AttrDict, is_float
from starter_code.log import log_string, format_log_string, TabularEnvManager


"""
* Algorithm level (epoch, episode, learning rate, etc)
* Monitoring level (wall clock time, etc)
* MDP level (returns, etc)
"""

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
    def __init__(self, agent, task_progression, rl_alg, logger, device, args):
        self.organism = agent
        self.task_progression = task_progression
        self.rl_alg = rl_alg
        self.logger = logger
        self.device = device
        self.args = args
        self.parallel_collect = False  #if this does not update state_histogram??
        self.logger.printf(self.organism)

        self.steps = 0
        self.min_return = np.inf
        self.max_return = -np.inf

    def collect_samples(self, epoch, env_manager):
        """
            Invariants:
                - metrics will always be updated after this method
                - this method will always collect self.rl_alg.num_samples_before_update samples
                - this method will not modify any state in the organism

            What this method does
                - returns stats from a bunch of rollouts
                - updates env_manager's logger based on stats
        """
        self.logger.printf('Collecting Samples...')
        t0 = time.time()
        gt.stamp('Epoch {}: Before Collect Samples'.format(epoch))
        collector = collect_train_samples_parallel if self.parallel_collect else collect_train_samples_serial

        self.organism.to('cpu')
        stats_collector = collector(
            epoch=epoch, 
            max_steps=self.rl_alg.num_samples_before_update, 
            objects=dict(
                max_episode_length=env_manager.max_episode_length,
                env=env_manager.env,
                stats_collector_builder=self.stats_collector_builder, 
                sampler_builder=self.exploration_sampler_builder, 
                organism=self.organism,
                seed=self.args.seed,
                printer=self.logger.printf,
                ),
            )
        self.organism.to(self.device)
        t1 = time.time()

        stats = stats_collector.bundle_batch_stats()
        t2 = time.time()

        for episode_data in stats_collector.data['episode_datas']:
            for e in episode_data:
                self.organism.store_transition(e)
        t3 = time.time()

        self.logger.printf('Bundle time: {}'.format(t2-t1))
        self.logger.printf('Storage time: {}'.format(t3-t2))

        ######################################################
        # update metrics
        self.steps += stats['total_steps']
        if stats['min_return'] < self.min_return:
            self.min_return = stats['min_return']
        if stats['max_return'] > self.max_return:
            self.max_return = stats['max_return']
        ######################################################

        gt.stamp('Epoch {}: After Collect Samples'.format(epoch))

        self.logger.printf('Epoch {}: Time to Collect Samples: {}'.format(epoch, time.time()-t0))
        return stats

    def main_loop(self, max_epochs):
        # populate replay buffer before training
        for epoch in gt.timed_for(range(max_epochs)):
            if epoch % self.args.eval_every == 0:
                self.eval_step(epoch)  # somehow this is not deterministic for decentralized?
            self.train_step(epoch)
        self.finish_training()

    def train_step(self, epoch):
        assert len(self.task_progression) == 1
        train_env_manager = self.task_progression.sample(i=epoch, mode='train')

        epoch_stats = self.collect_samples(epoch, train_env_manager)
        if epoch % self.args.log_every == 0:
            self.logger.printf(format_log_string(self.log(epoch, epoch_stats, mode='train')))
        if epoch % self.args.eval_every == 0:
            # this ticks the epoch and steps
            self.visualize(train_env_manager, epoch, epoch_stats, self.logger.expname)
        if epoch % self.args.save_every == 0:
            self.save(train_env_manager, epoch, epoch_stats)
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

    def log(self, epoch, epoch_stats, mode):
        s = log_string(OrderedDict({
            '{} epoch'.format(mode): epoch,
            'env steps this batch': epoch_stats['total_steps'],
            'env steps taken': self.steps,
            'avg return': epoch_stats['mean_return'],
            'std return': epoch_stats['std_return'],
            'min return': epoch_stats['min_return'],
            'max return': epoch_stats['max_return'],
            'min return ever': self.min_return,
            'max return ever': self.max_return
            }))
        return [s]

    def test(self, epoch, env_manager, num_test):
        stats_collector = self.stats_collector_builder()
        for i in range(num_test):
            with torch.no_grad():
                evaluation_sampler = self.evaluation_sampler_builder(self.organism)
                env_manager.env.seed(int(1e8)*self.args.seed+epoch)
                episode_data = evaluation_sampler.sample_episode(
                    env=env_manager.env, 
                    max_steps_this_episode=env_manager.max_episode_length,
                    render=True)
                ##########################################
                if epoch % self.args.visualize_every == 0:
                    if i == 0 and self.organism.discrete:
                        self.get_qualitative_output(
                            env_manager, evaluation_sampler, episode_data, epoch, i)
                        # if env_manager.visual:
                        #     # you could imagine having the sampler also return the best episode
                        #     # or perhaps you could bundle thiis in the stats_collector

                        #     bids = evaluation_sampler.get_bids_for_episode(episode_data)
                        #     returns = sum([e.reward for e in episode_data])
                        #     frames = [e.frame for e in episode_data]

                        #     env_manager.save_video(epoch, i, bids, returns, frames)

                ##########################################
            stats_collector.append(episode_data, eval_mode=True)
        stats = stats_collector.bundle_batch_stats(eval_mode=True)
        return stats

    def get_qualitative_output(self, env_manager, evaluation_sampler, episode_data, epoch, i):
        if env_manager.visual:
            # you could imagine having the sampler also return the best episode
            # or perhaps you could bundle thiis in the stats_collector

            bids = evaluation_sampler.get_bids_for_episode(episode_data)
            returns = sum([e.reward for e in episode_data])
            frames = [e.frame for e in episode_data]

            env_manager.save_video(epoch, i, bids, returns, frames)
        elif isinstance(env_manager, TabularEnvManager):
            """
            sorted:

            | time t | state | action | winner | bids | payoffs |
            | time t+1 | state | action | winner | bids | payoffs |
            """ 
            for t, step_data in enumerate(episode_data):
                step_dict = OrderedDict(
                    t='{}\t'.format(t),
                    state=env_manager.env.from_onehot(step_data.state),
                    action=step_data.action,
                    next_state=env_manager.env.from_onehot(step_data.state),
                    reward='{}\t'.format(step_data.reward),
                    mask=step_data.mask,
                    winner=step_data.winner,
                    bids=', '.join(['{}: {:.5f}'.format(a_id, bid) for a_id, bid in step_data.bids.items()]),
                    payoffs=', '.join(['{}: {:.5f}'.format(a_id, payoff) for a_id, payoff in step_data.payoffs.items()]),)
                self.logger.printf(log_string(step_dict))


    def update_metrics(self, env_manager, epoch, stats, metrics):
        """
            Assumes that we update every whole number of epochs

            Purpose: append to a list of things you will plot

        """
        env_manager.update_variable(name='epoch', index=epoch, value=epoch)
        env_manager.update_variable(name='steps', index=epoch, value=self.steps)

        for metric in metrics:
            env_manager.update_variable(
                name=metric, index=epoch, value=stats[metric], include_running_avg=True)

        # just literally save to CSV right now

    def plot_metrics(self, env_manager, name, metrics):
        env_manager.plot(
            var_pairs=[(('steps', k)) for k in metrics],
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
        metrics = ['min_return', 'max_return', 'mean_return', 'std_return',
                       'min_steps', 'max_steps', 'mean_steps', 'std_steps']
        self.update_metrics(env_manager, epoch, stats, metrics)
        self.plot_metrics(env_manager, name, metrics)

    def eval_step(self, epoch):
        self.logger.printf('Evaluating...')
        self.organism.to('cpu')
        for env_manager in self.task_progression[epoch]['test']:
            t0 = time.time()
            stats = self.test(epoch, env_manager, num_test=self.args.num_test)
            t1 = time.time()
            self.logger.printf(format_log_string(self.log(epoch, stats, mode='eval')))
            self.visualize(env_manager, epoch, stats, env_manager.env_name, eval_mode=True)
            t2 = time.time()
            self.save(env_manager, epoch, stats)
            t3 = time.time()

            self.logger.printf('Time to sample test examples: {}'.format(t1-t0))
            self.logger.printf('Time to visualize: {}'.format(t2-t1))
            self.logger.printf('Time to save: {}'.format(t3-t2))
        self.organism.to(self.device)


class CentralizedExperiment(Experiment):
    def __init__(self, agent, task_progression, rl_alg, logger, device, args):
        super(CentralizedExperiment, self).__init__(agent, task_progression, rl_alg, logger, device, args)
        self.exploration_sampler_builder = exploration_sampler_builder
        self.evaluation_sampler_builder = evaluation_sampler_builder
        self.stats_collector_builder = Centralized_RL_Stats

def exploration_sampler_builder(organism):
    return Sampler(
            organism=organism,
            eval_mode=False, 
            step_info=AgentStepInfo, 
            deterministic=False, 
            )

def evaluation_sampler_builder(organism):
    return Sampler(
            organism=organism,
            eval_mode=True, 
            step_info=AgentStepInfo, 
            deterministic=False, 
            )



