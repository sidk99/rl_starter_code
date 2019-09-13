import cv2
from collections import defaultdict
import numpy as np
import torch

from log import RunningAverage

import ipdb
from tqdm import tqdm

class Experiment():
    def __init__(self, agent, env_manager, rl_alg, logger, device, args):
        self.agent = agent
        self.env_manager = env_manager
        self.rl_alg = rl_alg
        self.logger = logger
        self.device = device
        self.args = args

    def render(self, env, scale):
        frame = env.render(mode='rgb_array')
        h, w = frame.shape[:-1]
        frame = cv2.resize(frame, dsize=(int(h*scale), int(w*scale)), interpolation=cv2.INTER_CUBIC)
        return frame

    def get_bids_for_episode(self, episode_data):
        episode_bids = defaultdict(lambda: [])
        for step in episode_data:
            probs = list(step['action_dist'].probs.detach()[0].cpu().numpy())
            for index, prob in enumerate(probs):
                episode_bids[index].append(prob)
        return episode_bids

    def sample_trajectory(self, deterministic, render):
        episode_data = []
        state = self.env_manager.env.reset()
        #################################################
        # Debugging MiniGrid
        if type(state) == dict:
            state = state['image']
        #################################################
        for t in range(self.rl_alg.max_buffer_size):  # Don't infinite loop while learning
            state_var = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_dict = self.agent(state_var, deterministic=deterministic)
            next_state, reward, done, _ = self.env_manager.env.step(action_dict['action'])
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
                frame = self.render(env=self.env_manager.env, scale=0.25)# width=600, height=400)
                e['frame'] = frame
            episode_data.append(e)
            self.agent.store_transition(e)
            if done:
                break
            state = next_state
        stats = {'return': sum([e['reward'] for e in episode_data]),
                 'steps': t,
                 'actions': [e['action'] for e in episode_data]}
        return episode_data, stats

    def collect_samples(self, deterministic):
        num_steps = 0
        all_episodes_data = []

        while num_steps < self.rl_alg.max_buffer_size:
            episode_data, episode_stats= self.sample_trajectory(deterministic, False)
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
            if i_episode % self.args.eval_every == 0:
                stats = self.test(i_episode=i_episode, max_episodes=10)
                self.logger.saver.save(i_episode, 
                    {'args': self.args,
                     'experiment': stats,
                     'agent': self.agent.get_state_dict()})

            episode_data, stats = self.collect_samples(deterministic=False)
            ret = stats['avg_return'] # but we shouldn't do a running avg actually
            running_return = run_avg.update_variable('reward', ret)

            def update_optimizer_lr(optimizer, scheduler, name):
                before_lr = optimizer.state_dict()['param_groups'][0]['lr']
                scheduler.step()
                after_lr = optimizer.state_dict()['param_groups'][0]['lr']
                to_print_alr = 'Learning rate for {} was {}. Now it is {}.'.format(name, before_lr, after_lr)
                if before_lr != after_lr:
                    to_print_alr += ' Learning rate changed!'
                    self.logger.printf(to_print_alr)

            if i_episode >= self.args.anneal_policy_lr_after:
                update_optimizer_lr(
                    optimizer=self.agent.policy_optimizer,
                    scheduler=self.agent.po_scheduler,
                    name='policy')
                update_optimizer_lr(
                    optimizer=self.agent.value_optimizer,
                    scheduler=self.agent.vo_scheduler,
                    name='value')

            if i_episode % self.args.update_every == 0:
                self.rl_alg.improve(self.agent)

            if i_episode % self.args.log_every == 0:
                self.logger.printf('Episode {}\tAvg Return: {:.2f}\tMin Return: {:.2f}\tMax Return: {:.2f}\tRunning Return: {:.2f}'.format(
                    i_episode, stats['avg_return'], stats['min_return'], stats['max_return'], running_return))

    def test(self, i_episode, max_episodes):
        visualize = True
        returns = []
        for i in tqdm(range(max_episodes)):
            with torch.no_grad():
                # TODO: something about this being deterministic is making things worse. Why?
                episode_data, stats = self.sample_trajectory(deterministic=False, render=visualize)
                ret = stats['return']

                if i == 0 and visualize and self.agent.policy.discrete:
                    bids = self.get_bids_for_episode(episode_data)
                    self.env_manager.save_video(i_episode, i, bids, ret, episode_data)
            returns.append(ret)
        returns = np.array(returns)
        stats = {'batch': i_episode,
                 'mean_return': np.mean(returns),
                 'std_return': np.std(returns),
                 'min_return': np.min(returns),
                 'max_return': np.max(returns)}
        self.env_manager.update_variable(name='i_episode', index=i_episode, value=i_episode)
        for metric in ['min_return', 'max_return', 'mean_return', 'std_return']:
            self.env_manager.update_variable(
                name=metric, index=i_episode, value=stats[metric], include_running_avg=True)
        self.env_manager.plot(
            var_pairs=[(('i_episode', k)) for k in ['min_return', 'max_return', 'mean_return', 'std_return']],
            expname=self.logger.expname)
        self.logger.pprintf(stats)
        return stats