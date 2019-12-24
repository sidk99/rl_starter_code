from collections import defaultdict
import numpy as np
import pickle
import time
import torch
import starter_code.env_utils as eu
from starter_code.utils import AttrDict, from_np, to_np
from starter_code.filter import MeanStdFilter, NoFilter

import multiprocessing as mp

def collect_train_samples_serial(epoch, max_steps, objects, pid=0, queue=None):
    env = objects['env']
    stats_collector = objects['stats_collector_builder']()
    sampler = objects['sampler_builder'](objects['organism'])

    env.seed(1000000+pid)
    torch.manual_seed(1000000+pid)
    np.random.seed(1000000+pid)

    start = time.time()
    num_steps = 0
    while num_steps < max_steps:
        max_timesteps_this_episode = min(max_steps - num_steps, objects['max_episode_length'])

        episode_data = sampler.sample_episode(
            env=env,
            max_timesteps_this_episode=max_timesteps_this_episode)

        stats_collector.append(episode_data)
        num_steps += len(episode_data)
    end = time.time()
    print('PID: {} Time to collect samples: {}'.format(pid, end-start))

    if queue is not None:
        queue.put([pid, stats_collector.data])
    else:
        return stats_collector

def collect_train_samples_parallel(epoch, max_steps, objects):
    num_workers = 8
    num_steps_per_worker = max_steps // num_workers
    num_residual_steps = max_steps - num_steps_per_worker * num_workers

    # you should initialize workers right here
    queue = mp.Queue()
    workers = []
    for i in range(num_workers):
        worker_steps = num_steps_per_worker + num_residual_steps if i == 0 else num_steps_per_worker
        worker_kwargs = dict(
            epoch=epoch,
            max_steps=worker_steps,
            objects=objects,  # should I copy these?
            pid=i+1,
            queue=queue)
        workers.append(mp.Process(target=collect_train_samples_serial, kwargs=worker_kwargs))
    for j, worker in enumerate(workers):
        worker.start()

    start = time.time()
    master_stats_collector = objects['stats_collector_builder']()
    for j, worker in enumerate(workers):
        worker_pid, worker_stats_data = queue.get()
        master_stats_collector.extend(worker_stats_data)
    end = time.time()
    print('Time to extend master_stats_collector: {}'.format(end-start))

    for j, worker in enumerate(workers):
        worker.join()

    assert master_stats_collector.get_total_steps() == max_steps
    return master_stats_collector

class BasicStepInfo():
    def __init__(self, state, action_dict, next_state, mask, reward):
        self.state = state
        self.action_dict = action_dict
        self.next_state = next_state
        self.mask = mask
        self.reward = reward

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __str__(self):
        return str(self.__dict__)

    def to_dict(self):
        return self.__dict__

class AgentStepInfo(BasicStepInfo):
    def __init__(self, state, action_dict, next_state, mask, reward):
        super(AgentStepInfo, self).__init__(
            state, action_dict, next_state, mask, reward)
        self.action = self.action_dict['stored_action']
        self.action_dist = self.action_dict['action_dist']

class Sampler():
    """
        one sampler for exploration
        one sampler for evaluation
    """
    def __init__(self, organism, eval_mode, step_info, deterministic):#, render):
        self.organism = organism
        self.deterministic = deterministic
        self.step_info_builder = step_info
        self.eval_mode = eval_mode

    def sample_timestep(self, env, organism, state, render=False):
        state_var = from_np(state, 'cpu')
        with torch.no_grad():
            action_dict = organism.forward(state_var, deterministic=self.deterministic)
        next_state, reward, done, _ = env.step(action_dict['action'])
        mask = 0 if done else 1
        e = self.step_info_builder(
            state=state,
            action_dict=action_dict,
            next_state=next_state,
            mask=mask,
            reward=reward)
        if render:
            e.frame = eu.render(env=env, scale=0.25)
        return next_state, done, e

    def begin_episode(self):
        state = self.env.reset()
        return state

    def finish_episode(self, episode_data):
        return episode_data

    def get_bids_for_episode(self, episode_data):
        episode_bids = defaultdict(lambda: [])
        for step in episode_data:
            probs = list(to_np(step['action_dist'].probs)[0])
            for index, prob in enumerate(probs):
                episode_bids[index].append(prob)
        return episode_bids

    def sample_episode(self, env, max_timesteps_this_episode, render=False):
        # or can set self.env and self.organism here
        ###################################
        # Dangerous? only if you modify them in begin_episode or finish_episode
        self.env = env
        ###################################
        episode_data = []
        state = self.begin_episode()
        for t in range(max_timesteps_this_episode):
            state, done, e = self.sample_timestep(env, self.organism, state, render)
            episode_data.append(e)
            if done: break
        if not done:
            assert t == max_timesteps_this_episode-1 
            # save the environment state here
        episode_info = self.finish_episode(episode_data)
        return episode_info

# # the purpose is that this is just a big fat datastructure that you can subclass for your needs
class Compound_RL_Stats:
    def __init__(self):
        self.reset()

    # initialize
    def reset(self):
        self.data = dict(
            returns=[],
            steps=[],
            episode_datas=[]
            )

    # update
    def append(self, episode_data, eval_mode=False):
        self.data['returns'].append(sum([e.reward for e in episode_data]))
        self.data['steps'].append(len(episode_data))
        self.data['episode_datas'].append(episode_data)

    def extend(self, other_data):
        self.data['returns'].extend(other_data['returns'])
        self.data['steps'].extend(other_data['steps'])
        self.data['episode_datas'].extend(other_data['episode_datas'])

    def get_total_steps(self):
        return sum(self.data['steps'])

    # query
    def bundle_batch_stats(self, eval_mode=False):
        stats = dict(num_episodes=len(self.data['steps']))
        stats = dict({**stats, **self.log_metrics(np.array(self.data['returns']), 'return')})
        stats = dict({**stats, **self.log_metrics(np.array(self.data['steps']), 'steps')})
        return stats

    def __len__(self):
        return len(self.data['steps'])

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

# what's the purpose? what's the scope?
# the purpose: for data collection during training (what about testing?)
# the scope is a single batch of collected samples. You overwrite the data every time you collect a new batch
class Centralized_RL_Stats(Compound_RL_Stats):
    def __init__(self):
        super(Centralized_RL_Stats, self).__init__()