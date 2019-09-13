import copy
import csv
import datetime
import imageio
import glob
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import shutil
import pprint
import torch
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import heapq


def create_logdir(root, dirname, setdate):
    logdir = os.path.join(root, dirname)
    if setdate:
        if not dirname == '': logdir += '-'
        logdir += '{date:%Y-%m-%d_%H-%M-%S}'.format(
        date=datetime.datetime.now())
    os.mkdir(logdir)
    return logdir

class RunningAverage(object):
    def __init__(self):
        super(RunningAverage, self).__init__()
        self.data = {}
        self.alpha = 0.01

    def update_variable(self, key, value):
        self.data[key] = value # overwrite
        if 'running_'+key not in self.data:
            self.data['running_'+key] = value
        else:
            self.data['running_'+key] = (1-self.alpha) * self.data['running_'+key] + self.alpha * value
        return copy.deepcopy(self.data['running_'+key])

    def get_value(self, key):
        if 'running_'+key in self.data:
            return self.data['running_'+key]
        else:
            assert KeyError

    def get_last_value(self, key):
        if key in self.data:
            return self.data[key]
        else:
            assert KeyError

class Saver(object):
    def __init__(self, checkpoint_dir, heapsize=1):
        self.checkpoint_dir = checkpoint_dir
        self.heapsize = heapsize
        self.most_recents = []  # largest is most recent
        self.bests = []  # largest is best
        self.create_save_file = lambda i_episode: os.path.join(self.checkpoint_dir, 'ckpt_batch{}.pth.tar'.format(i_episode))

        with open(os.path.join(self.checkpoint_dir, 'summary.csv'), 'w') as f:
            csv_writer = csv.DictWriter(f, fieldnames=['recent', 'best'])
            csv_writer.writeheader()

    def save(self, i_episode, state_dict):
        ckpt_id = int(state_dict['experiment']['batch'])
        ckpt_return = float(state_dict['experiment']['mean_return'])
        ckpt_name = self.create_save_file(i_episode)
        heapq.heappush(self.most_recents, (ckpt_id, ckpt_name))
        heapq.heappush(self.bests, (ckpt_return, ckpt_name))
        torch.save(state_dict, ckpt_name)
        # self.evict()
        self.save_summary()
        print('Saved to {}.'.format(ckpt_name))

    def save_summary(self):
        most_recent = os.path.basename(heapq.nlargest(1, self.most_recents)[0][-1])
        best = os.path.basename(heapq.nlargest(1, self.bests)[0][-1])
        with open(os.path.join(self.checkpoint_dir, 'summary.csv'), 'a') as f:
            csv_writer = csv.DictWriter(f, fieldnames=['recent', 'best'])
            csv_writer.writerow({'recent': most_recent, 'best': best})

    def evict(self):
        to_evict = set()
        if len(self.most_recents) > self.heapsize:
            most_recents = [x[-1] for x in heapq.nlargest(1, self.most_recents)]
            higest_returns = [x[-1] for x in heapq.nlargest(1, self.bests)]

            least_recent = heapq.heappop(self.most_recents)[-1]
            lowest_return = heapq.heappop(self.bests)[-1]

            # only evict least_recent if it is not in highest_returns
            if least_recent not in higest_returns and os.path.exists(least_recent):
                os.remove(least_recent)
            # only evict lowest_return if it is not in lowest_return
            if lowest_return not in most_recents and os.path.exists(lowest_return):
                os.remove(lowest_return)


class MultiBaseLogger(object):
    def __init__(self, args):
        self.args = args
        self.root = args.root
        self.expname = args.expname
        self.logdir = create_logdir(root=self.root, dirname=self.expname, setdate=True)
        self.checkpoint_dir = create_logdir(root=self.logdir, dirname='checkpoints', setdate=False)
        self.saver = Saver(self.checkpoint_dir)

    def get_state_dict(self):
        return {'logdir': self.logdir, 'checkpoint_dir': self.checkpoint_dir}

    def printf(self, string):
        if self.args.printf:
            f = open(os.path.join(self.logdir, self.expname+'.txt'), 'a')
            print(string, file=f)
        else:
            print(string)

    def pprintf(self, string):
        if self.args.printf:
            f = open(os.path.join(self.logdir, self.expname+'.txt'), 'a')
            pprint.pprint(string, stream=f)
        else:
            pprint.pprint(string)

    def remove_logdir(self):
        should_remove = input('Remove {}? [y/n] '.format(self.logdir))
        if should_remove == 'y':
            shutil.rmtree(self.logdir)
            print('Removed {}'.format(self.logdir))
        else:
            print('Did not remove {}'.format(self.logdir))

class EnvLogger(object):
    def __init__(self, args):
        super(EnvLogger, self).__init__()
        self.data = {}
        self.metrics = {}
        self.run_avg = RunningAverage()

    def get_state_dict(self):
        state_dict = {
            **super(EnvLogger, self).get_state_dict(),
            **{'qualitative_dir': self.qualitative_dir, 'quantitative_dir': self.quantitative_dir}}
        return state_dict

    def set_logdir(self, logdir):
        self.logdir = logdir
        self.qualitative_dir = create_logdir(root=self.logdir, dirname='qualitative', setdate=False)
        self.quantitative_dir = create_logdir(root=self.logdir, dirname='quantitative', setdate=False)

    def add_variable(self, name, incl_run_avg=False, metric=None):
        self.data[name] = []
        if incl_run_avg:
            self.data['running_{}'.format(name)] = []
        if metric is not None:
            self.add_metric(
                name='running_{}'.format(name), 
                initial_val=metric['value'], 
                comparator=metric['cmp'])

    def update_variable(self, name, index, value, include_running_avg=False):
        if include_running_avg:
            running_name = 'running_{}'.format(name)
            self.run_avg.update_variable(name, value)
            self.data[running_name].append((index, self.run_avg.get_value(name)))
        self.data[name].append((index, value))

    def get_recent_variable_value(self, name):
        index, recent_value = copy.deepcopy(self.data[name][-1])
        return recent_value

    def has_running_avg(self, name):
        return self.run_avg.exists(name)

    def add_metric(self, name, initial_val, comparator):
        self.metrics[name] = {'value': initial_val, 'cmp': comparator}

    def plot(self, var_pairs, expname):
        for var1_name, var2_name in var_pairs:
            x_indices, x_values = zip(*self.data[var1_name])
            y_indices, y_values = zip(*self.data[var2_name])
            fname = '{}_{}'.format(expname, var2_name)
            plt.plot(x_values,y_values)
            plt.xlabel(var1_name)
            plt.ylabel(var2_name)
            plt.savefig(os.path.join(self.quantitative_dir, '{}.png'.format(fname)))
            plt.close()

class EnvManager(EnvLogger):
    def __init__(self, env_name, args):
        super(EnvManager, self).__init__(args)
        self.env_name = env_name
        self.env_type = 'mg'  # CHANGED
        self.env = gym.make(env_name)  # CHANGED
        self.env.seed(args.seed)  # this should be args.seed
        self.initialize()

    def initialize(self):
        self.add_variable('i_episode')

class VisualEnvManager(EnvManager):
    def __init__(self, env_name, args):
        super(VisualEnvManager, self).__init__(env_name, args)

    def initialize(self):
        super(VisualEnvManager, self).initialize()
        self.add_variable('min_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('max_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('mean_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('std_return', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})

    def save_image(self, fname, i, ret, frame, bids_t):
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.grid(False)
        ax1.set_title('Timestep: {}'.format(i))
        ax1.imshow(frame)

        ax2 = fig.add_subplot(122)
        agent_ids, bids = zip(*bids_t)
        ax2.bar(agent_ids, bids, align='center', alpha=0.5)
        ax2.set_xticks(range(len(agent_ids)), map(int, agent_ids))
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel('Bid')
        ax2.set_xlabel('Action')
        fig.suptitle('Return: {}'.format(ret))  # can say what the return is here
        plt.tight_layout(pad=3)
        plt.savefig(os.path.join(self.qualitative_dir, fname))
        plt.close()

    def save_gif(self, prefix, gifname, i_episode, test_example, remove_images):
        def get_key(fname):
            basename = os.path.basename(fname)
            delimiter = '_t'
            start = basename.rfind(delimiter)
            key = int(basename[start+len(delimiter):-len('.png')])
            return key
        fnames = sorted(glob.glob('{}/{}_e{}_n{}_t*.png'.format(self.qualitative_dir, prefix, i_episode, test_example)), key=get_key)
        images = [imageio.imread(fname) for fname in fnames]
        imshape = images[0].shape
        for pad in range(2):
            images.append(imageio.core.util.Array(np.ones(imshape).astype(np.uint8)*255))
        imageio.mimsave(os.path.join(self.qualitative_dir, gifname), images)
        if remove_images:
            for fname in fnames:
                os.remove(fname)

    def save_video(self, i_episode, test_example, bids, ret, society_episode_data):
        frames = [e['frame'] for e in society_episode_data]
        for i, frame in tqdm(enumerate(frames)):
            fname = '{}_e{}_n{}_t{}.png'.format(self.env_name, i_episode, test_example, i)
            agent_ids = sorted(bids.keys())
            bids_t = [(agent_id, bids[agent_id][i]) for agent_id in agent_ids]
            self.save_image(fname, i, ret, frame, bids_t)
        gifname = 'vid{}_{}_{}.gif'.format(self.env_name, i_episode, test_example)
        self.save_gif(self.env_name, gifname, i_episode, test_example, remove_images=True)

class GymEnvManager(VisualEnvManager):
    def __init__(self, env_name, args):
        super(GymEnvManager, self).__init__(env_name, args)
        self.state_dim = self.env.observation_space.shape[0]
        self.is_disc_action = len(self.env.action_space.shape) == 0
        self.action_dim = self.env.action_space.n if self.is_disc_action else self.env.action_space.shape[0]

class MinigridEnvManager(VisualEnvManager):
    def __init__(self, env_name, args):
        super(MinigridEnvManager, self).__init__(env_name, args)
        full_state_dim = self.env.observation_space.spaces['image'].shape  # (H, W, C)
        self.state_dim = full_state_dim[:-1]  # (H, W)
        self.is_disc_action = len(self.env.action_space.shape) == 0
        self.action_dim = self.env.action_space.n if self.is_disc_action else self.env.action_space.shape[0]