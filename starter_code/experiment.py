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

import starter_code.env_utils as eu

import starter_code.utils as u

def analyze_size(obj, obj_name):
    obj_pickle = pickle.dumps(obj)
    print('Size of {}: {}'.format(obj_name, sys.getsizeof(obj_pickle)))

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
            e = dict(
                state=state,
                action=action_dict['stored_action'],
                action_dist=action_dict['action_dist'],
                mask=mask,
                reward=reward)
            if render:
                frame = eu.render(env=env, scale=0.25)
                e['frame'] = frame
            episode_data.append(e)
            self.organism.store_transition(e)
            if done:
                break
            state = next_state
        stats = dict(
            returns=sum([e['reward'] for e in episode_data]),
            moves=t+1,
            actions=[e['action'] for e in episode_data])
        episode_info = dict(
            organism_episode_data=episode_data,
            episode_stats=stats)
        return episode_info

    def collect_samples(self, deterministic):
        num_steps = 0
        num_episodes = 0
        all_returns = []
        all_moves = []

        while num_steps < self.rl_alg.max_buffer_size:
            train_env_manager = self.task_progression.sample(i=self.epoch, mode='train')
            episode_info = self.sample_episode(env=train_env_manager.env, deterministic=deterministic, render=False)
            all_returns.append(episode_info['episode_stats']['returns'])
            all_moves.append(episode_info['episode_stats']['moves'])
            num_steps += (episode_info['episode_stats']['moves'])
            num_episodes += 1
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
            num_episodes=num_episodes
            )
        self.run_avg.update_variable('reward', stats['mean_return'])

        episode_info['epoch_stats'] = stats
        return episode_info

    # @gt.wrap
    def train(self, max_epochs):
        # populate replay buffer before training

        import time

        for epoch in gt.timed_for(range(max_epochs)):
            if epoch % self.args.eval_every == 0:
                stats = self.eval(epoch=epoch)  # this needs to be done for multiple environments.

                for mode in ['train']:
                    for env_manager in self.task_progression[self.epoch][mode]:
                        env_manager.saver.save(epoch, 
                            {'args': self.args,
                             'epoch': epoch,
                             'logger': self.logger.get_state_dict(),
                             'experiment': stats[env_manager.env_name],
                             'organism': self.organism.get_state_dict()},
                             self.logger.printf)

            gt.stamp('Epoch {}: Before Collect Samples'.format(epoch))
            epoch_info = self.collect_samples(deterministic=False)
            gt.stamp('Epoch {}: After Collect Samples'.format(epoch))

            if epoch % self.args.eval_every == 0:
                metrics = ['min_return', 'max_return', 'mean_return', 'std_return',
                       'min_moves', 'max_moves', 'mean_moves', 'std_moves']
                self.logger.update_variable(name='epoch', index=epoch, value=epoch)
                # this should also be based on the running reward.
                self.update_metrics(self.logger, metrics, epoch, epoch_info['epoch_stats'])
                self.plot_metrics(self.logger, metrics, self.logger.expname)

                self.logger.saver.save(epoch, 
                    {'args': self.args,
                     'epoch': epoch,
                     'logger': self.logger.get_state_dict(),
                     'experiment': epoch_info['epoch_stats'],
                     'organism': self.organism.get_state_dict()},
                     self.logger.printf)

            if epoch >= self.args.anneal_policy_lr_after:
                self.organism.step_optimizer_schedulers(self.logger.printf)

            gt.stamp('Epoch {}: Before Update'.format(epoch))
            t0 = time.time()
            self.organism.update(self.rl_alg)
            self.logger.printf('Epoch {}: After Update: {}'.format(epoch, time.time()-t0))
            gt.stamp('Epoch {}: After Update'.format(epoch))

            self.organism.clear_buffer()

            if epoch % self.args.log_every == 0:
                self.log(epoch, epoch_info)

        self.finish_training()

    def finish_training(self):
        # env.close()
        # plt.close()
        print(gt.report())
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
                ret = episode_info['episode_stats']['returns']
                mov = episode_info['episode_stats']['moves']
                if i == 0 and visualize and self.organism.discrete:
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

    def update_metrics(self, env_manager, metrics, epoch, stats):
        for metric in metrics:
            env_manager.update_variable(
                name=metric, index=epoch, value=stats[metric], include_running_avg=True)

    def plot_metrics(self, env_manager, metrics, name):
        env_manager.plot(
            var_pairs=[(('epoch', k)) for k in metrics],
            expname=name,
            pfunc=self.logger.printf)

    def eval(self, epoch):
        metrics = ['min_return', 'max_return', 'mean_return', 'std_return',
                   'min_moves', 'max_moves', 'mean_moves', 'std_moves']
        multi_task_stats = {} 
        for mode in ['train']:
            for env_manager in self.task_progression[self.epoch][mode]:
                stats = self.test(epoch, env_manager, num_test=self.args.num_test, visualize=True)

                env_manager.update_variable(name='epoch', index=epoch, value=epoch)
                self.update_metrics(env_manager, metrics, epoch, stats)
                self.plot_metrics(env_manager, metrics, env_manager.env_name)

                self.logger.pprintf(stats)
                multi_task_stats[env_manager.env_name] = stats
        return multi_task_stats


