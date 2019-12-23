import numpy as np
from starter_code.log import create_logdir

class TaskNode(object):
    def __init__(self, task_name, parents=None):
        self.task_name 
        self.parents = parents  # a list of TaskNodes

# you will save this task node in the checkpoint

class TaskDistribution(object):
    """
        Contains a set of K environment managers
    """
    def __init__(self):
        """
            the input argument has to be immutable!
            if you make environment_managers=[] it might NOT initialize the object with an empty list!
        """
        self.environment_managers = []

    def __iter__(self):
        for env_manager in self.environment_managers:
            yield env_manager

    def __len__(self):
        return len(self.environment_managers)

    def __str__(self):
        return '{}\n'.format(type(self)) + '\n'.join('\t{}: {}'.format(e.env_name, e) for e in self.environment_managers)

    def get_canonical(self):
        return self.environment_managers[0]

    def append(self, environment_manager):
        self.environment_managers.append(environment_manager)
        assert self.consistent_spec()

    @property
    def state_dim(self):
        return self.get_canonical().state_dim

    @property
    def action_dim(self):
        return self.get_canonical().action_dim

    @property
    def is_disc_action(self):
        return self.get_canonical().is_disc_action

    def consistent_spec(self):
        same_state_dim = [e.state_dim == self.environment_managers[0].state_dim for e in self.environment_managers]
        same_action_dim = [e.action_dim == self.environment_managers[0].action_dim for e in self.environment_managers]
        return same_state_dim and same_action_dim

    def sample(self):
        return np.random.choice(self.environment_managers)


class TaskDistributionGroup(object):
    """
        Contains a group of train/val/test task distributions
    """
    def __init__(self):  # could add a val distribution
        self.dists = {}

    def __setitem__(self, mode, dist):
        self.dists[mode] = dist
        assert self.consistent_spec()

    def __getitem__(self, mode):
        return self.dists[mode]

    def __str__(self):
        return '{}\n'.format(type(self)) + '\n'.join(['\t{}: {}'.format(k, str(v)) for k, v in self.dists.items()])

    def get_canonical(self):
        return list(self.dists.values())[0]

    @property
    def state_dim(self):
        return self.get_canonical().state_dim

    @property
    def action_dim(self):
        return self.get_canonical().action_dim

    @property
    def is_disc_action(self):
        return self.get_canonical().is_disc_action

    def consistent_spec(self):
        return all([v.consistent_spec() for v in self.dists.values()])  # incorrect. need to check state_dim and action_dim

    def sample(self, mode):
        return self.dists[mode].sample()


class TaskProgression(object):
    """
        Contains a time series of TaskDistributionGroups
    """
    def __init__(self):
        self.task_dist_group_series = []

    def __getitem__(self, i):
        idx = min(i, len(self.task_dist_group_series)-1)  # keep last one if saturated
        return self.task_dist_group_series[idx]

    def __str__(self):
        return '{}\n'.format(type(self)) + '\n'.join('\t'.format(str(g)) for g in self.task_dist_group_series)

    def get_canonical(self):
        return self.task_dist_group_series[0]

    @property
    def state_dim(self):
        return self.get_canonical().state_dim

    @property
    def action_dim(self):
        return self.get_canonical().action_dim

    @property
    def is_disc_action(self):
        return self.get_canonical().is_disc_action

    def append(self, task_dist_group):
        self.task_dist_group_series.append(task_dist_group)
        assert self.consistent_spec()

    def consistent_spec(self):
        return all([v.consistent_spec() for v in self.task_dist_group_series])  # incorrect. need to check state_dim and action_dim

    def sample(self, i, mode):
        prog_group_id = min(i, len(self.task_dist_group_series)-1)  # keep last one if saturated
        return self.task_dist_group_series[prog_group_id].sample(mode)

def default_task_prog_spec(env_name):
    spec = {0: dict(train=[env_name], test=[env_name])}
    return spec

def task_prog_spec_multi(env_names):
    spec = {0: dict(train=env_names, test=env_names)}
    return spec

def construct_task_progression(task_prog_spec, env_manager_builder, logger, env_registry, args):
    """
    should be able to convert:
    {
        0: {
            train: ['MiniGrid-Empty-Random-5x5-v0', 'MiniGrid-Empty-5x5-v0'],
            test: ['MiniGrid-Empty-Random-6x6-v0', 'MiniGrid-Empty-6x6-v0'],
        },
        1: {
            train: ['MiniGrid-SimpleCrossingS9N1-v0', 'MiniGrid-SimpleCrossingS9N2-v0'],
            test: ['MiniGrid-SimpleCrossingS9N3-v0', 'MiniGrid-SimpleCrossingS11N5-v0'],
        },
    }
    to a TaskProgression object

    The directory structure is
        exp_root
            0
                train_folder_0_0
                train_folder_0_1
                test_folder_0_0
                test_folder_0_1
            1
                train_folder_1_0
                train_folder_1_1
                test_folder_1_0
                test_folder_1_1
    """
    task_progression = TaskProgression()
    for i, group in task_prog_spec.items():
        group_dir = create_logdir(
            root=logger.logdir, dirname='group_{}'.format(i), setdate=False)
        task_distribution_group = TaskDistributionGroup()
        for mode, envs in group.items():
            task_distribution = TaskDistribution()
            for env_name in envs:
                env_manager = env_manager_builder(env_name, env_registry, args)
                env_manager.set_logdir(create_logdir(
                    root=group_dir, 
                    dirname='{}_{}_{}'.format(env_name, i, mode), 
                    setdate=False))
                task_distribution.append(env_manager)
            task_distribution_group[mode] = task_distribution
        task_progression.append(task_distribution_group)
    return task_progression



#######
# debug


def debug_task_progression_mg():
    args = process_args(parse_args())

    env_registry = EnvRegistry()

    task_prog_spec = {
        0: {
            'train': ['MiniGrid-Empty-Random-5x5-v0', 'MiniGrid-Empty-5x5-v0'],
            'test': ['MiniGrid-Empty-Random-6x6-v0', 'MiniGrid-Empty-6x6-v0'],
        },
        1: {
            'train': ['MiniGrid-SimpleCrossingS9N1-v0', 'MiniGrid-SimpleCrossingS9N2-v0'],
            'test': ['MiniGrid-SimpleCrossingS9N3-v0', 'MiniGrid-SimpleCrossingS11N5-v0'],
        },
    }

    task_progression = construct_task_progression(task_prog_spec, GymEnvManager, args)

    for i in range(2):
        for mode in ['train', 'test']:
            for k in range(5):
                env_manager = task_progression.sample(i, mode)
                print('i: {}\tmode: {}\tk: {}\tenv_name: {}'.format(i, mode, k, env_manager.env_name), end=' ')
                # sample episode from this env_manager
                state = env_manager.env.reset()
                for t in range(10):
                    print('{:.2f}'.format(np.linalg.norm(state['image'])), end=' ')
                    action = np.random.randint(6)
                    next_state, reward, done, _ = env_manager.env.step(action)
                    state = next_state
                print('')

def debug_task_progression_tab():
    args = process_args(parse_args())

    env_registry = EnvRegistry()
    task_prog_spec = {
        0: {
            'train': ['3s2t2af', '3s2t2af'],
            'test': ['3s2t2af', '3s2t2af'],
        },
        1: {
            'train': ['3s2t2af', '3s2t2af'],
            'test': ['3s2t2af', '3s2t2af'],
        },
    }

    task_progression = construct_task_progression(task_prog_spec, TabularEnvManager, args)

    for i in range(2):
        for mode in ['train', 'test']:
            for k in range(5):
                env_manager = task_progression.sample(i, mode)
                print('i: {}\tmode: {}\tk: {}\tenv_name: {}'.format(i, mode, k, env_manager.env_name), end=' ')
                # sample episode from this env_manager
                state = env_manager.env.reset()
                for t in range(2):
                    print('{:.2f}'.format(np.linalg.norm(state)), end=' ')
                    action = np.random.randint(2)
                    next_state, reward, done, _ = env_manager.env.step(action)
                    state = next_state
                print('')
