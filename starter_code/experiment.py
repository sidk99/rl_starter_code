import cv2
from collections import defaultdict
import gtimer as gt
import numpy as np
import pickle
import sys
import torch

from log import RunningAverage

import ipdb
from tqdm import tqdm
import time

import starter_code.env_utils as eu

import starter_code.utils as u

def analyze_size(obj, obj_name):
    obj_pickle = pickle.dumps(obj)
    print('Size of {}: {}'.format(obj_name, sys.getsizeof(obj_pickle)))

class Sampler():
    """
        one sampler for exploration
        one sampler for evaluation
    """
    def __init__(self, deterministic, render, device):
        self.deterministic = deterministic
        self.render = render
        self.device = device

        # you need something to accumulate the stats


    # def reset_episode_info(self):
    #     self.episode_info = dict()

    def sample_timestep(self, env, organism, state):
        state_var = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_dict = organism.forward(state_var, deterministic=self.deterministic)
        next_state, reward, done, _ = env.step(action_dict['action'])
        mask = 0 if done else 1
        e = dict(
            state=state,
            action=action_dict['stored_action'],
            action_dist=action_dict['action_dist'],
            next_state=next_state,
            mask=mask,
            reward=reward)
        if self.render:
            frame = eu.render(env=env, scale=0.25)
            e['frame'] = frame
        return next_state, done, e

    def sample_episode(self, env, organism, max_timesteps_this_episode):
        episode_data = []
        state = env.reset()
        done = False

        for t in range(max_timesteps_this_episode):
            print('t: {}'.format(t))
            state, done, e = self.sample_timestep(
                env, organism, state)
            episode_data.append(e)
            organism.store_transition(e)
            if done:
                print('done')
                break
        if not done:
            # the only reason why broke the loop
            assert t == max_timesteps_this_episode-1 
            # save the environment state here

        stats = dict(
            returns=sum([e['reward'] for e in episode_data]),
            moves=t+1,
            actions=[e['action'] for e in episode_data])
        episode_info = dict(
            organism_episode_data=episode_data,
            episode_stats=stats)

        return episode_info

    def sample_many_episodes(self, env_manager):
        pass

    def save_env_state(self, env_manager):
        pass

    def load_env_state(self, env_manager):
        pass


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
        self.metrics = ['min_return', 'max_return', 'mean_return', 'std_return',
                       'min_moves', 'max_moves', 'mean_moves', 'std_moves']
        self.exploration_sampler = Sampler(
            deterministic=False, render=False, device=device)
        self.evaluation_sampler = Sampler(
            deterministic=False, render=True, device=device)

    def get_bids_for_episode(self, episode_info):
        episode_bids = defaultdict(lambda: [])
        for step in episode_info['organism_episode_data']:
            probs = list(step['action_dist'].probs.detach()[0].cpu().numpy())
            for index, prob in enumerate(probs):
                episode_bids[index].append(prob)
        return episode_bids

    ################################################################
    def sample_episode(self, env, max_timesteps_this_episode, deterministic, render):
        raise NotImplementedError
    ################################################################


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

            # if it makes it up here, that means max_timesteps_this_episode >= 
            max_timesteps_this_episode = min(
                self.rl_alg.num_samples_before_update - num_steps,
                train_env_manager.max_episode_length)

            print('num_steps: {} num_episodes: {}, max_timesteps_this_episode: {}, max_episode_length_from_env: {}'.format(num_steps, num_episodes, max_timesteps_this_episode, train_env_manager.max_episode_length))


            ################################################################
            episode_info = self.exploration_sampler.sample_episode(
                env=train_env_manager.env, 
                organism=self.organism,
                max_timesteps_this_episode=max_timesteps_this_episode)

            # episode_info = self.sample_episode(train_env_manager.env, max_timesteps_this_episode, deterministic=False, render=False)


    # def sample_episode(self, env, max_timesteps_this_episode, deterministic, render, record_payoffs=True, record_bids=False):  # record_bids will be informed by the env_manager based on the env_type
    #     episode_info = self.exploration_sampler.sample_episode(env, self.organism, max_timesteps_this_episode, deterministic, render, record_payoffs, record_bids)
    #     return episode_info

            
            ################################################################

            all_returns.append(episode_info['episode_stats']['returns'])
            all_moves.append(episode_info['episode_stats']['moves'])
            num_steps += (episode_info['episode_stats']['moves'])
            num_episodes += 1
        print('num_steps collected: {}'.format(num_steps))

        stats = dict(
            mean_return=np.mean(all_returns),
            min_return=np.min(all_returns),
            max_return=np.max(all_returns),
            std_return=np.std(all_returns),
            total_return=np.sum(all_returns),
            mean_moves=np.mean(all_moves),
            std_moves=np.std(all_moves),
            min_moves=np.min(all_moves),
            max_moves=np.max(all_moves),
            num_episodes=num_episodes,
            total_moves=num_steps,
            )
        self.run_avg.update_variable('reward', stats['mean_return'])

        episode_info['epoch_stats'] = stats
        print('num_steps: {} num_episodes: {}'.format(num_steps, num_episodes))
        gt.stamp('Epoch {}: After Collect Samples'.format(epoch))
        return episode_info

    def train(self, max_epochs):
        # populate replay buffer before training
        for epoch in gt.timed_for(range(max_epochs)):
            print('epoch: {}'.format(epoch))
            if epoch % self.args.eval_every == 0:
                stats = self.eval(epoch=epoch)

            epoch_info = self.collect_samples(epoch)
            
            if epoch % self.args.log_every == 0:
                self.log(epoch, epoch_info)

            if epoch % self.args.visualize_every == 0:
                self.visualize(self.logger, epoch, epoch_info['epoch_stats'], self.logger.expname)
            
            if epoch % self.args.save_every == 0:
                self.save(self.logger, epoch, epoch_info['epoch_stats'])

            self.update(epoch)

        self.finish_training()


    def finish_training(self):
        # env.close()
        # plt.close()
        # print(gt.report())
        pass


    def update(self, epoch):
        if epoch >= self.args.anneal_policy_lr_after:
            self.organism.step_optimizer_schedulers(self.logger.printf)

        gt.stamp('Epoch {}: Before Update'.format(epoch))
        t0 = time.time()
        self.organism.update(self.rl_alg)
        self.logger.printf('Epoch {}: After Update: {}'.format(epoch, time.time()-t0))
        gt.stamp('Epoch {}: After Update'.format(epoch))
        self.organism.clear_buffer()

    def log(self, epoch, epoch_info):
        stats = epoch_info['epoch_stats']
        self.logger.printf('Episode {}\tAvg Return: {:.2f}\tMin Return: {:.2f}\tMax Return: {:.2f}\tRunning Return: {:.2f}'.format(
            epoch, stats['mean_return'], stats['min_return'], stats['max_return'], self.run_avg.get_value('reward')))

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

                # episode_info = self.sample_episode(env_manager.env, env_manager.max_episode_length, deterministic=False, render=True)
                ################################################################

                ret = episode_info['episode_stats']['returns']
                mov = episode_info['episode_stats']['moves']

                if i == 0 and self.organism.discrete:
                    bids = self.get_bids_for_episode(episode_info)
                    env_manager.save_video(epoch, i, bids, ret, episode_info['organism_episode_data'])
            returns.append(ret)
            moves.append(mov)


        returns = np.array(returns)
        moves = np.array(moves)
        stats = {}
        stats = {**stats, **self.log_metrics(np.array(returns), 'return')}
        stats = {**stats, **self.log_metrics(np.array(moves), 'moves')}
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



