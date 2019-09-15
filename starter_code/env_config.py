from collections import namedtuple

import gym
import gym_minigrid

def dict_transpose(d):
    """
        input: {
            A: {
                B: X
            }
        }

        output: {
            B: {
                A
            }
        }
    """
    count = 0
    tranposed_dict = {}
    for key, subdict in d.items():
        for subkey in subdict:
            tranposed_dict[subkey] = key
            count += 1
    assert len(tranposed_dict.keys()) == count
    return tranposed_dict

class EnvRegistry():
    def __init__(self):
        self.envs_type_name = {
            'gym': {
                'CartPole-v0',
            },
            'mg': {
                'MiniGrid-MultiRoom-N6-v0',
                'MiniGrid-MultiRoom-N2-S4-v0',
                'MiniGrid-MultiRoom-N4-S5-v0',
                'MiniGrid-MultiRoom-N6-v0',
                'MiniGrid-Empty-Random-5x5-v0',
                'MiniGrid-Empty-5x5-v0',
                'MiniGrid-Empty-Random-6x6-v0',
                'MiniGrid-Empty-6x6-v0',
                'MiniGrid-Empty-Random-7x7-v0',
                'MiniGrid-Empty-Random-8x8-v0',
                'MiniGrid-Empty-Random-9x9-v0',

                'MiniGrid-SimpleCrossingS9N1-v0',
                'MiniGrid-SimpleCrossingS9N2-v0',
                'MiniGrid-SimpleCrossingS9N3-v0',
                'MiniGrid-SimpleCrossingS11N5-v0',
            }
        }
        self.typecheck(self.envs_type_name)
        self.envs_name_type = dict_transpose(self.envs_type_name)

    def typecheck(self, d):
        assert type(d) == dict
        for key, value in d.items():
            assert type(value) == dict or type(value) == set

    def get_env_constructor(self, env_name):
        return lambda: gym.make(env_name)

    def get_env_type(self, env_name):
        return self.envs_name_type[env_name]