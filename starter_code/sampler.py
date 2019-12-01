from collections import defaultdict
import numpy as np
import torch
import starter_code.env_utils as eu
from starter_code.utils import AttrDict, from_np, to_np

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

    # def __getattr__(self, name):
    #     return self.__dict__[key]

    # def __setattr__(self, name, value):
    #     self.__dict__[name] = value

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
    def __init__(self, step_info, deterministic, render, device):
        self.deterministic = deterministic
        self.render = render
        self.device = device
        self.step_info_builder = step_info

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

    def begin_episode(self):
        self.episode_data = []
        state = self.env.reset()
        return state

    def finish_episode(self):
        for e in self.episode_data:
            self.organism.store_transition(e)
        ##########################################
        episode_info = AttrDict(
            returns=sum([e.reward for e in self.episode_data]),
            steps=len(self.episode_data),
            )
        if self.render:
            episode_info.frames = [e.frame for e in self.episode_data]
            episode_info.bids = self.get_bids_for_episode(self.episode_data)

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
        episode_info = self.finish_episode()
        return episode_info

    def sample_many_episodes(self, env_manager):
        pass

    def save_env_state(self, env_manager):
        pass

    def load_env_state(self, env_manager):
        pass


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
    def append(self, episode_info):
        self.data.returns.append(episode_info.returns)
        self.data.steps.append(episode_info.steps)
        self.data.episode_infos.append(episode_info)

    # query
    def bundle_batch_stats(self):
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