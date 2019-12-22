from collections import defaultdict
import numpy as np
import time
import torch
import starter_code.env_utils as eu
from starter_code.utils import AttrDict, from_np, to_np
from starter_code.filter import MeanStdFilter, NoFilter

import torch.multiprocessing as mp


def collect_train_samples_serial(epoch, max_steps, objects):
    task_progression = objects.task_progression
    stats_collector = objects.stats_collector_builder()
    sampler = objects.sampler_builder()
    organism = objects.organism

    num_steps = 0
    # maybe have a memory object here?
    episode_num = 0

    while num_steps < max_steps:
        print('Episode {}'.format(episode_num))
        train_env_manager = task_progression.sample(i=epoch, mode='train')
        max_timesteps_this_episode = min(max_steps - num_steps, train_env_manager.max_episode_length)
        episode_info = sampler.sample_episode(
            env=train_env_manager.env, 
            organism=organism,
            max_timesteps_this_episode=max_timesteps_this_episode)

        # episode_info = sampler.dummy_method(
        #     env=train_env_manager.env, 
        #     organism=organism,
        #     max_timesteps_this_episode=max_timesteps_this_episode)

        stats_collector.append(episode_info)
        num_steps += (episode_info.steps)
        episode_num += 1
        print('Steps so far: {}'.format(num_steps))

    stats = stats_collector.bundle_batch_stats()
    assert num_steps == stats['total_steps'] == max_steps
    return stats


# def collect_train_samples_parallel(epoch, max_steps, objects):
#     num_steps = 0
#     num_workers = 8
#     stats_collector = stats_collector_builder()
#     num_steps_per_worker = max_steps // num_workers
#     num_residual_steps = max_steps - num_steps_per_worker * num_workers

#     # you should initialize workers right here
#     workers = None
#     all_worker_stats = []
#     for i, worker in enumerate(workers):
#         worker_steps = num_steps_per_worker + num_residual_steps if i == 0 else num_steps_per_worker
#         worker_stats = collect_train_samples_serial(epoch, worker_steps, copy.deepcopy(objects))
#         all_worker_stats.append(worker_stats)

#     # wait for all workers to finish joining
#     stats_collector.extend(all_worker_stats)
#     stats = stats_collector.bundle_batch_stats()
#     assert stats['total_steps'] == max_steps
#     return stats

def collect_train_samples_parallel(epoch, max_steps, objects):
    num_steps = 0
    num_workers = 8
    stats_collector = stats_collector_builder()
    num_steps_per_worker = max_steps // num_workers
    num_residual_steps = max_steps - num_steps_per_worker * num_workers

    # you should initialize workers right here
    queue = mp.Queue()
    workers = []
    for i in range(num_workers):
        worker_steps = num_steps_per_worker + num_residual_steps if i == 0 else num_steps_per_worker
        worker_kwargs = dict(
            pid=i+1,
            queue=queue,
            epoch=epoch,
            max_steps=worker_steps,
            objects=objects)
        workers.append(mp.Process(target=collect_train_samples_serial, kwargs=worker_kwargs))
    for worker in workers:
        worker.start()

    all_worker_stats = []
    for worker in workers:
        worker_stats = queue.get()
        all_worker_stats.append(worker_stats)

    stats = all_worker_stats

    # processes = []
    # all_worker_stats = []
    # for i, worker in enumerate(workers):
    #     worker_steps = num_steps_per_worker + num_residual_steps if i == 0 else num_steps_per_worker
    #     worker_stats = collect_train_samples_serial(epoch, worker_steps, copy.deepcopy(objects))
    #     all_worker_stats.append(worker_stats)

    # # wait for all workers to finish joining
    # stats_collector.extend(all_worker_stats)
    # stats = stats_collector.bundle_batch_stats()
    # assert stats['total_steps'] == max_steps
    return stats

    # def update(self, rl_alg):
    #     learnable_active_agents = [a for a in self.get_active_agents() if a.learnable]
    #     if self.args.parallel_update:
    #         processes = []
    #         for agent in learnable_active_agents:
    #             p = mp.Process(target=agent_update, args=(agent, rl_alg))
    #             p.start()
    #             processes.append(p)
    #         for p in processes:
    #             p.join()
    #     else:
    #         # watch out here because different agents' replay buffers may be different sizes
    #         for agent in learnable_active_agents: 
    #             agent.update(rl_alg)


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

class AgentStepInfo(BasicStepInfo):
    def __init__(self, state, action_dict, next_state, mask, reward):
        super(AgentStepInfo, self).__init__(
            state, action_dict, next_state, mask, reward)
        self.action = self.action_dict.stored_action
        self.action_dist = self.action_dict.action_dist


class Sampler():
    """
        one sampler for exploration
        one sampler for evaluation
    """
    def __init__(self, eval_mode, obs_dim, step_info, deterministic, render, device):
        self.deterministic = deterministic
        self.render = render
        self.device = device
        self.step_info_builder = step_info
        self.eval_mode = eval_mode

    def sample_timestep(self, env, organism, state):
        state_var = from_np(state, self.device)
        with torch.no_grad():
            action_dict = organism.forward(state_var, deterministic=self.deterministic)
        next_state, reward, done, _ = env.step(action_dict.action)
        mask = 0 if done else 1
        e = self.step_info_builder(
            state=state,
            action_dict=action_dict,
            next_state=next_state,
            mask=mask,
            reward=reward)
        if self.render:
            e.frame = eu.render(env=env, scale=0.25)
        return next_state, done, e

    def dummy_method(self, env, organism, max_timesteps_this_episode):
        time.sleep(1)
        return {i: np.random.random() for i in range(10)}

    def begin_episode(self):
        self.episode_data = []
        state = self.env.reset()
        return state

    def finish_episode(self):
        start = time.time()
        if not self.eval_mode:
            for e in self.episode_data:
                self.organism.store_transition(e)
        after_store_transition = time.time()

        episode_info = AttrDict(
            returns=sum([e.reward for e in self.episode_data]),
            steps=len(self.episode_data),
            )
        if self.render:
            episode_info.frames = [e.frame for e in self.episode_data]
            episode_info.bids = self.get_bids_for_episode(self.episode_data)

        after_episode_info = time.time()

        print('\tTime to store transition: {}'.format(after_store_transition-start))
        print('\tTime to create episode info: {}'.format(after_episode_info-after_store_transition))
        return episode_info

    def get_bids_for_episode(self, episode_data):
        episode_bids = defaultdict(lambda: [])
        for step in episode_data:
            probs = list(to_np(step.action_dist.probs)[0])
            for index, prob in enumerate(probs):
                episode_bids[index].append(prob)
        return episode_bids

    def record_episode_data(self, e):
        self.episode_data.append(e)

    def sample_episode(self, env, organism, max_timesteps_this_episode):
        start = time.time()
        # or can set self.env and self.organism here
        ###################################
        # Dangerous? only if you modify them in begin_episode or finish_episode
        self.env = env
        self.organism = organism
        ###################################
        state = self.begin_episode()
        for t in range(max_timesteps_this_episode):
            state, done, e = self.sample_timestep(env, organism, state)
            self.record_episode_data(e)
            if done:
                break
        if not done:
            assert t == max_timesteps_this_episode-1 
            # save the environment state here
        after_sampling = time.time()
        episode_info = self.finish_episode()
        after_finish = time.time()
        print('Time to sample episode: {}'.format(after_sampling-start))
        print('Time to finish episode: {}'.format(after_finish-after_sampling))
        return episode_info


class ParallelSampler(Sampler):
    def __init__(self, step_info, deterministic, render, device):
        super(ParallelSampler, self).__init__(step_info, deterministic, render, device)

    def sample_episode(self, env, organism, max_timesteps_this_episode):
        # now make copies of each organism and a copy of the environment

        # maybe use ray
        cloned_episode_infos = []
        for cloned_env, cloned_organism in clones:
            cloned_episode_info = super(ParallelSampler, self).sample_episode(
                cloned_env, cloned_organism, max_timesteps_this_episode)
            cloned_episode_infos.append(cloned_episode_info)
        return cloned_episode_infos


# the purpose is that this is just a big fat datastructure that you can subclass for your needs
class Compound_RL_Stats:
    def __init__(self):
        self.reset()

    # initialize
    def reset(self):
        self.data = AttrDict(
            returns=[],
            steps=[],
            episode_infos=[])

    # update
    def append(self, episode_info, eval_mode=False):
        self.data.returns.append(episode_info.returns)
        self.data.steps.append(episode_info.steps)
        self.data.episode_infos.append(episode_info)

    # query
    def bundle_batch_stats(self, eval_mode=False):
        stats = dict(num_episodes=len(self.data.steps))
        stats = dict({**stats, **self.log_metrics(np.array(self.data.returns), 'return')})
        stats = dict({**stats, **self.log_metrics(np.array(self.data.steps), 'steps')})
        return stats

    def __len__(self):
        return len(self.data.steps)

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