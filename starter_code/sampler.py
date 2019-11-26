import torch
import starter_code.env_utils as eu

def from_np(np_array, device):
    return torch.tensor([np_array]).float().to(device)

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
        state_var = from_np(state, self.device)

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