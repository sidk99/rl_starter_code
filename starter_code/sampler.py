from collections import defaultdict
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

        # you need something to accumulate the stats

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

    def begin_episode(self, env):
        self.episode_data = []
        state = env.reset()
        return state

    def run_episode(self):
        pass

    def finish_episode(self, organism):
        # 1. organism record_episode_data
        # 2. bundle stats (I'm not even sure we need this)

        for e in self.episode_data:
            organism.store_transition(e)

        episode_info = AttrDict(
            returns=sum([e.reward for e in self.episode_data]),
            moves=len(self.episode_data),
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

    def sample_episode(self, env, organism, max_timesteps_this_episode):
        state = self.begin_episode(env)

        for t in range(max_timesteps_this_episode):
            state, done, e = self.sample_timestep(
                env, organism, state)
            self.episode_data.append(e)
            if done:
                print('done')
                break
        if not done:
            assert t == max_timesteps_this_episode-1 
            # save the environment state here

        episode_info = self.finish_episode(organism)
        return episode_info

    def sample_many_episodes(self, env_manager):
        pass

    def save_env_state(self, env_manager):
        pass

    def load_env_state(self, env_manager):
        pass