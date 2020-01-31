from collections import namedtuple
import gym
import gym_minigrid
from gym_minigrid.wrappers import ImgObsWrapper
import babyai
import pprint

from gym.wrappers.time_limit import TimeLimit


from starter_code.envs import OneStateOneStepKActionEnv, OneHotSquareGridWorldK, OneHotGridWorldK, OneHotChainK

"""
    Minigrid reward range: (0,1)
    CartPole reward range: (-inf, inf)
"""


class EnvInfo():
    def __init__(self, env_name, env_type, reward_shift=0, reward_scale=1, **kwargs):
        self.env_name = env_name
        self.env_type = env_type

        # just set default
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale

        for key, value in kwargs.items():
            self.__setattr__(key, value)



    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __str__(self):
        return str(self.__dict__)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return str(self.__dict__)


def build_env_infos(d):
    """
        input: {
            A: {
                B: X
            }
        }

        output: {
            B: {
                A: X
            }
        }
    """
    count = 0
    tranposed_dict = dict()
    for key, subdict in d.items():
        for subkey in subdict:
            # tranposed_dict[subkey] = key
            tranposed_dict[subkey] = EnvInfo(env_name=subkey, env_type=key, **subdict[subkey])
            count += 1
    assert len(tranposed_dict.keys()) == count
    return tranposed_dict

def simplify_name(names):
    return '_'.join(''.join(x for  x in name if not x.islower()) for name in names)


"""
note that return scale depends on both the gamma and the max reward


1. normalize individual rewards to be withini [0, 1]
2. given gamma, bound the total return.

"""

class RewardNormalize(gym.RewardWrapper):
    def __init__(self, env, scale=1, shift=0):
        super().__init__(env)
        self.scale = scale
        self.shift = shift

    def reward(self, r):
        return (r - self.shift) * self.scale

class GymRewardNormalize(RewardNormalize):
    def __init__(self, env, scale=1, shift=0):
        if isinstance(env, TimeLimit):
            self._max_episode_steps = env._max_episode_steps
            # unwrap the TimeLimit
            RewardNormalize.__init__(self, env.env, scale, shift)  
        else:
            assert False

class MiniGridRewardNormalize(RewardNormalize):
    def __init__(self, env, scale=1, shift=0):
        RewardNormalize.__init__(self, env, scale, shift)
        self.max_steps = env.max_steps


class EnvRegistry():
    def __init__(self):
        self.atari_envs = {
            'Breakout-ram-v4': dict(),
            'Pong-ram-v4': dict(),
            'SpaceInvaders-ram-v4': dict(),
            'Enduro-ram-v4': dict(),

            'Breakout-ramNoFrameskip-v4': dict(),
            'Pong-ramNoFrameskip-v4': dict(),
            'SpaceInvaders-ramNoFrameskip-v4': dict(),
            'Enduro-ramNoFrameskip-v4': dict(),
        }


        self.envs_type_name = {
            'gym': {
                'CartPole-v0': dict(reward_shift=0, reward_scale=1.0/(200*(1-0))),  # 200 steps * ([1 max] - [0 min])
                'InvertedPendulum-v2': dict(),
                'HalfCheetah-v2': dict(),
                # 'LunarLander-v2': dict(reward_shift=-100, reward_scale=1.0/(1000*(100--100))),  # 1000 steps * ([100 max] - [-100 min]) which seems to be empirically the range. --> This causes it to be between 0 and 0.5

                # 'LunarLander-v2': dict(reward_shift=0, reward_scale=1.0/(500*(100--100))),  # 1000 steps * ([100 max] - [-100 min]) which seems to be empirically the range. --> This causes it to be between 0 and 0.5


                'LunarLander-v2': dict(reward_shift=0, reward_scale=1.0/250),  # DONE_I_THINKs

                'MountainCar-v0': dict(),
                'Acrobot-v1': dict(),
                'Taxi-v2': dict(),
                'FrozenLake-v0': dict(),
            },
            'mg': {
                'MiniGrid-MultiRoom-N6-v0': dict(),
                'MiniGrid-MultiRoom-N2-S4-v0': dict(),
                'MiniGrid-MultiRoom-N4-S5-v0': dict(),
                'MiniGrid-MultiRoom-N6-v0': dict(),
                'MiniGrid-Empty-5x5-v0': dict(),
                'MiniGrid-Empty-Random-6x6-v0': dict(),
                'MiniGrid-Empty-6x6-v0': dict(),
                'MiniGrid-Empty-Random-7x7-v0': dict(),
                'MiniGrid-Empty-Random-8x8-v0': dict(),
                'MiniGrid-Empty-Random-9x9-v0': dict(),

                'MiniGrid-SimpleCrossingS9N1-v0': dict(),
                'MiniGrid-SimpleCrossingS9N2-v0': dict(),
                'MiniGrid-SimpleCrossingS9N3-v0': dict(),
                'MiniGrid-SimpleCrossingS11N5-v0': dict(),

                'MiniGrid-Unlock-v0': dict(),
                'MiniGrid-Empty-Random-5x5-v0': dict(reward_scale=0.9),  # done; TODO: make this a function of the horizon length though, which it so happens is 100. But in general this is not True.
                'MiniGrid-MultiRoom-N2-S4-v0': dict(),

                # Expanding Action Set
                'BabyAI-PickupKey-v0': dict(reward_scale=1.0),  # 64 steps
                'BabyAI-OpenDoorDebug-v0': dict(reward_scale=1.0/2),  # 576 steps DONE_I_THINK



                'BabyAI-OpenOneDoor-v0': dict(reward_scale=0.8),



            },
            'tab': {
                # 0. purpose: one-arm bandit; simplest case of regressing to Q-value; sanity check
                '1S1T1A': dict(constructor=lambda: OneStateOneStepKActionEnv(1)),

                # 1. purpose: k-arm bandit: observe what happens as the number of agents grows
                '1S1T2A': dict(constructor=lambda: OneStateOneStepKActionEnv(2)),
                '1S1T3A': dict(constructor=lambda: OneStateOneStepKActionEnv(3)),
                '1S1T4A': dict(constructor=lambda: OneStateOneStepKActionEnv(4)),
                '1S1T9A': dict(constructor=lambda: OneStateOneStepKActionEnv(9)),

                # '3S2T2A': ThreeStateTwoStepTwoActionEnv,  # do this one for multi-step
                # '3S2T2AF': ThreeStateTwoStepTwoActionFineEnv,
                # '3S2T2AA': ThreeStateTwoStepTwoActionAsymmetricEnv,
                # '6S2T2A': SixStateTwoStepEnv,  # do this now
                # '6S2T2G': SixStateTwoStepFlippedEnv,
                # '2S1T2A': TwoStateOneStepTwoActionEnv,  # do this one

                'GW2': dict(constructor=lambda: OneHotSquareGridWorldK(2)),
                'GW3': dict(constructor=lambda: OneHotSquareGridWorldK(3)),
                'GW4': dict(constructor=lambda: OneHotSquareGridWorldK(4)),

                'GW2R': dict(constructor=lambda: OneHotSquareGridWorldK(2, rand_init=True)),
                'GW3R': dict(constructor=lambda: OneHotSquareGridWorldK(3, rand_init=True)),
                'GW4R': dict(constructor=lambda: OneHotSquareGridWorldK(4, rand_init=True)),

                'GW2RR': dict(constructor=lambda: OneHotSquareGridWorldK(2, rand_init=True, goal=None)),
                'GW3RR': dict(constructor=lambda: OneHotSquareGridWorldK(3, rand_init=True, goal=None)),
                'GW4RR': dict(constructor=lambda: OneHotSquareGridWorldK(4, rand_init=True, goal=None)),

                'GW2RG0': dict(constructor=lambda: OneHotSquareGridWorldK(2, rand_init=True, goal=0)),
                'GW2RG1': dict(constructor=lambda: OneHotSquareGridWorldK(2, rand_init=True, goal=1)),
                'GW2RG2': dict(constructor=lambda: OneHotSquareGridWorldK(2, rand_init=True, goal=2)),
                'GW2RG3': dict(constructor=lambda: OneHotSquareGridWorldK(2, rand_init=True, goal=3)),

                'CW2': dict(constructor=lambda eplencoeff, step_reward: OneHotChainK(
                    2, eplencoeff=eplencoeff, step_reward=step_reward)),
                'CW3': dict(constructor=lambda eplencoeff, step_reward: OneHotChainK(
                    3, eplencoeff=eplencoeff, step_reward=step_reward)),
                'CW4': dict(constructor=lambda eplencoeff, step_reward: OneHotChainK(
                    4, eplencoeff=eplencoeff, step_reward=step_reward)),
                'CW5': dict(constructor=lambda eplencoeff, step_reward: OneHotChainK(
                    5, eplencoeff=eplencoeff, step_reward=step_reward)),
                'CW6': dict(constructor=lambda eplencoeff, step_reward: OneHotChainK(
                    6, eplencoeff=eplencoeff, step_reward=step_reward)),
            }
        }

        self.envs_type_name['gym'] = {**self.envs_type_name['gym'], **self.atari_envs}

        self.typecheck(self.envs_type_name)
        self.env_infos = build_env_infos(self.envs_type_name)

    def typecheck(self, d):
        assert type(d) == dict
        for key, value in d.items():
            assert type(value) == dict or type(value) == set

    def get_env_constructor(self, env_name):
        env_type = self.get_env_type(env_name)
        if env_type == 'mg':
            constructor = lambda: MiniGridRewardNormalize(
                ImgObsWrapper(gym.make(env_name)), 
                scale=self.env_infos[env_name].reward_scale, 
                shift=self.env_infos[env_name].reward_shift)

        elif env_type == 'gym':
            constructor = lambda: GymRewardNormalize(
                gym.make(env_name), 
                scale=self.env_infos[env_name].reward_scale, 
                shift=self.env_infos[env_name].reward_shift)
        elif env_type == 'tab':
            """
            Here you should explicitly design the reward structure
            """
            constructor = self.env_infos[env_name].constructor
        else:
            assert False

        return constructor

    def get_env_type(self, env_name):
        return self.env_infos[env_name].env_type

    def get_reward_normalization_info(self, env_name):
        env_info = self.env_infos[env_name]
        return dict(reward_shift=env_info.reward_shift, reward_scale=env_info.reward_scale)



