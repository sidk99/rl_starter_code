import copy
from collections import defaultdict
import csv
import datetime
import imageio
import glob
import gym
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import pandas as pd
import pprint
import shutil
import torch
from matplotlib.ticker import MaxNLocator
import heapq
from env_config import EnvRegistry
from starter_code.utils import is_float


er = EnvRegistry()

def mkdirp(logdir):
    try:
        os.makedirs(logdir)
    except FileExistsError:
        overwrite = 'o'
        while overwrite not in ['y', 'n']:
            overwrite = input('{} exists. Overwrite? [y/n] '.format(logdir))
        if overwrite == 'y':
            shutil.rmtree(logdir)
            os.mkdir(logdir)
        else:
            raise FileExistsError

def create_logdir(root, dirname, setdate):
    logdir = os.path.join(root, dirname)
    if setdate:
        if not dirname == '': logdir += '__'
        logdir += '{date:%Y-%m-%d_%H-%M-%S}'.format(
        date=datetime.datetime.now())
    mkdirp(logdir)
    return logdir


class RunningAverage(object):
    def __init__(self):
        super(RunningAverage, self).__init__()
        self.data = {}

    def update_variable(self, key, value):
        self.data[key] = value # overwrite
        if 'running_'+key not in self.data:
            self.data['running_'+key] = value
            self.data['counter_'+key] = 1
        else:
            self.data['counter_'+key] += 1
            n = self.data['counter_'+key]
            self.data['running_'+key] = float(n-1)/n * self.data['running_'+key] + 1.0/n * value

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

class ExponentialRunningAverage(RunningAverage):
    def __init__(self):
        super(ExponentialRunningAverage, self).__init__()
        self.data = {}
        self.alpha = 0.01

    def update_variable(self, key, value):
        self.data[key] = value # overwrite
        if 'running_'+key not in self.data:
            self.data['running_'+key] = value
        else:
            self.data['running_'+key] = (1-self.alpha) * self.data['running_'+key] + self.alpha * value

class Saver(object):
    def __init__(self, checkpoint_dir, heapsize=1):
        self.checkpoint_dir = checkpoint_dir
        self.heapsize = heapsize
        self.most_recents = []  # largest is most recent
        self.bests = []  # largest is best

        with open(os.path.join(self.checkpoint_dir, 'summary.csv'), 'w') as f:
            csv_writer = csv.DictWriter(f, fieldnames=['recent', 'best'])
            csv_writer.writeheader()

    def save(self, epoch, state_dict, pfunc):
        ckpt_id = epoch
        ckpt_return = float(state_dict['mean_return'])
        ckpt_name = os.path.join(
            self.checkpoint_dir, 'ckpt_batch{}.pth.tar'.format(epoch))
        heapq.heappush(self.most_recents, (ckpt_id, ckpt_name))
        heapq.heappush(self.bests, (ckpt_return, ckpt_name))
        torch.save(state_dict, ckpt_name)
        self.save_summary()
        pfunc('Saved to {}.'.format(ckpt_name))

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

class BaseLogger(object):
    def __init__(self, args):
        super(BaseLogger, self).__init__()
        self.data = {}
        self.metrics = {}
        self.run_avg = RunningAverage()

    # you should have a method that restores the self.data from the csv

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
        """
            Appends to the log
        """
        if include_running_avg:
            running_name = 'running_{}'.format(name)
            self.run_avg.update_variable(name, value)
            self.data[running_name].append((index, self.run_avg.get_value(name)))
        self.data[name].append((index, value))

    def get_recent_variable_value(self, name):
        index, recent_value = self.data[name][-1]
        return recent_value

    def has_running_avg(self, name):
        return self.run_avg.exists(name)

    def add_metric(self, name, initial_val, comparator):
        self.metrics[name] = {'value': initial_val, 'cmp': comparator}

    def plot(self, var_pairs, expname, pfunc):  # EXPNAME
        self.save_csv(expname, pfunc)
        self.plot_from_csv(
            var_pairs=var_pairs,
            expname=expname)
        self.clear_data()

    def save_csv(self, expname, pfunc):
        csv_dict = defaultdict(dict)
        for key, value in self.data.items():
            for index, e in value:
                csv_dict[index][key] = e
        filename = os.path.join(self.quantitative_dir,'{}.csv'.format(expname))  # DIR
        pfunc('Saving to {}'.format(filename))
        file_exists = os.path.isfile(filename)
        with open(filename, 'a+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.data.keys())
            if not file_exists:
                writer.writeheader()
            for i in sorted(csv_dict.keys()):
                writer.writerow(csv_dict[i])

    def load_csv(self, expname):
        filename = os.path.join(self.quantitative_dir,'{}.csv'.format(expname))  # DIR
        df = pd.read_csv(filename)
        return df

    def plot_from_csv(self, var_pairs, expname):
        df = self.load_csv(expname)
        for var1_name, var2_name in var_pairs:
            data = df[[var1_name, var2_name]].dropna()
            x = data[var1_name].tolist()
            y = data[var2_name].tolist()
            fname = '{}_{}'.format(expname, var2_name)
            plt.plot(x,y)
            plt.xlabel(var1_name)
            plt.ylabel(var2_name)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.savefig(os.path.join(self.quantitative_dir,'_csv{}.png'.format(fname)))  # DIR
            plt.close()

    def clear_data(self):
        for key in self.data:
            self.data[key] = []

class MultiBaseLogger(BaseLogger):
    def __init__(self, args):
        super(MultiBaseLogger, self).__init__(args)
        self.args = args
        self.subroot = os.path.join('runs', args.subroot)
        self.expname = args.expname
        self.logdir = create_logdir(root=self.subroot, dirname=self.expname, setdate=True)

        self.qualitative_dir = create_logdir(root=self.logdir, dirname='qualitative', setdate=False)
        self.quantitative_dir = create_logdir(root=self.logdir, dirname='quantitative', setdate=False)

        self.checkpoint_dir = create_logdir(root=self.logdir, dirname='checkpoints', setdate=False)
        self.saver = Saver(self.checkpoint_dir)
        self.printf('Subroot: {}\nExperiment Name: {}\nLog Directory: {}\nCheckpoint Directory: {}'.format(
            self.subroot, self.expname, self.logdir, self.checkpoint_dir))

        self.code_dir = create_logdir(root=self.logdir, dirname='code', setdate=False)
        json.dump(vars(args), open(os.path.join(self.code_dir, 'params.json'), 'w'))

        self.initialize()

    def save_source_code(self):
        pass

    def initialize(self):
        self.add_variable('epoch')
        self.add_variable('steps')

        self.add_variable('min_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('max_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('mean_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('std_return', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})

        self.add_variable('min_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})
        self.add_variable('max_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})
        self.add_variable('mean_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})
        self.add_variable('std_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})

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

class EnvLogger(BaseLogger):
    def __init__(self, args):
        super(EnvLogger, self).__init__(args)

    def get_state_dict(self):
        state_dict = {
            **super(EnvLogger, self).get_state_dict(),
            **{'qualitative_dir': self.qualitative_dir, 'quantitative_dir': self.quantitative_dir}}
        return state_dict

    def set_logdir(self, logdir):
        self.logdir = logdir
        self.qualitative_dir = create_logdir(root=self.logdir, dirname='qualitative', setdate=False)
        self.quantitative_dir = create_logdir(root=self.logdir, dirname='quantitative', setdate=False)

        self.checkpoint_dir = create_logdir(root=self.logdir, dirname='checkpoints', setdate=False)
        self.saver = Saver(self.checkpoint_dir)
        print('Qualitative Directory: {}\nQuantitative Directory: {}\nLog Directory: {}\nCheckpoint Directory: {}'.format(self.qualitative_dir, self.quantitative_dir, self.logdir, self.checkpoint_dir))

class EnvManager(EnvLogger):
    def __init__(self, env_name, env_registry, args):
        super(EnvManager, self).__init__(args)
        self.env_name = env_name
        self.env_type = env_registry.get_env_type(env_name)
        self.env = env_registry.get_env_constructor(env_name)()
        self.visual = False  # default
        self.initialize()

    def initialize(self):
        self.add_variable('epoch')
        self.add_variable('steps')

        self.add_variable('min_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('max_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('mean_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('std_return', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})

        self.add_variable('min_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})
        self.add_variable('max_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})
        self.add_variable('mean_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})
        self.add_variable('std_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})


# class TabularEnvManager(EnvManager):
#     def __init__(self, env_name, env_registry, args):
#         super(TabularEnvManager, self).__init__(env_name, env_registry, args)
#         self.state_dim = self.env.state_dim
#         self.action_dim = len(self.env.actions)
#         self.starting_states = self.env.starting_states
#         self.max_episode_length = self.env.eplen

#     def initialize(self):
#         super(TabularEnvManager, self).initialize()
#         for state in self.env.starting_states:
#             self.add_variable('min_return_s{}'.format(state), incl_run_avg=True, 
#                 metric={'value': -np.inf, 'cmp': operator.ge})
#             self.add_variable('max_return_s{}'.format(state), incl_run_avg=True, 
#                 metric={'value': -np.inf, 'cmp': operator.ge})
#             self.add_variable('mean_return_s{}'.format(state), incl_run_avg=True, 
#                 metric={'value': -np.inf, 'cmp': operator.ge})
#             self.add_variable('std_return_s{}'.format(state), incl_run_avg=True, 
#                 metric={'value': np.inf, 'cmp': operator.le})

#             self.add_variable('min_steps_s{}'.format(state), incl_run_avg=True, 
#                 metric={'value': np.inf, 'cmp': operator.le})
#             self.add_variable('max_steps_s{}'.format(state), incl_run_avg=True, 
#                 metric={'value': np.inf, 'cmp': operator.le})
#             self.add_variable('mean_steps_s{}'.format(state), incl_run_avg=True, 
#                 metric={'value': np.inf, 'cmp': operator.le})
#             self.add_variable('std_steps_s{}'.format(state), incl_run_avg=True, 
#                 metric={'value': np.inf, 'cmp': operator.le})

#     # should I merge VickreyLogger with this?
#     # I think I should merge VickreyLogger with this.



class TabularEnvManager(EnvManager):
    def __init__(self, env_name, env_registry, args):
        super(TabularEnvManager, self).__init__(env_name, env_registry, args)
        self.state_dim = self.env.state_dim
        self.action_dim = len(self.env.actions)
        self.starting_states = self.env.starting_states
        self.max_episode_length = self.env.eplen

        ##################################################
        self.agent_data = {}
        ##################################################

    def initialize(self):
        super(TabularEnvManager, self).initialize()
        for state in self.env.starting_states:
            self.add_variable('min_return_s{}'.format(state), incl_run_avg=True, 
                metric={'value': -np.inf, 'cmp': operator.ge})
            self.add_variable('max_return_s{}'.format(state), incl_run_avg=True, 
                metric={'value': -np.inf, 'cmp': operator.ge})
            self.add_variable('mean_return_s{}'.format(state), incl_run_avg=True, 
                metric={'value': -np.inf, 'cmp': operator.ge})
            self.add_variable('std_return_s{}'.format(state), incl_run_avg=True, 
                metric={'value': np.inf, 'cmp': operator.le})

            self.add_variable('min_steps_s{}'.format(state), incl_run_avg=True, 
                metric={'value': np.inf, 'cmp': operator.le})
            self.add_variable('max_steps_s{}'.format(state), incl_run_avg=True, 
                metric={'value': np.inf, 'cmp': operator.le})
            self.add_variable('mean_steps_s{}'.format(state), incl_run_avg=True, 
                metric={'value': np.inf, 'cmp': operator.le})
            self.add_variable('std_steps_s{}'.format(state), incl_run_avg=True, 
                metric={'value': np.inf, 'cmp': operator.le})

    # should I merge VickreyLogger with this?
    # I think I should merge VickreyLogger with this.

    ##################################################
    def save_json(self, fname):
        with open(os.path.join(self.quantitative_dir, fname), 'w') as fp:
            ujson.dump(self.agent_data, fp, sort_keys=True, indent=4)

    def record_state_variable(self, state, step, step_dict, metric):
        """
        self.agent.data[metric][state][a_id][step] = value

        assume no agent dropout
        """
        if metric not in self.agent_data:
            self.agent_data[metric] = {}
        if state not in self.agent_data[metric]:
            self.agent_data[metric][state] = {}
        for a_key in step_dict:
            a_id = int(a_key[len('agent_'):])
            if a_id not in self.agent_data[metric][state]:
                self.agent_data[metric][state][a_id] = {}
            self.agent_data[metric][state][a_id][step] = step_dict[a_key][metric]

    def visualize_data(self, state, title, metric):
        for a_id in self.agent_data[metric][state]:
            data_indices, data_values = zip(*self.agent_data[metric][state][a_id].items())
            plt.plot(data_indices, data_values, label='{} for agent {}'.format(metric, a_id), color='C{}'.format(a_id))
        plt.legend()
        if 'bid' in metric:
            plt.ylim(-0.1, 1.1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.title(title)
        plt.tight_layout()
        fname = '{}_{}.png'.format(title.replace(' ', '_'), metric)
        plt.savefig(os.path.join(self.quantitative_dir, fname))
        plt.close()  # plt.clf() shrinks the plot for some reason

    def plot_bid_Q_differences(self, stats):
        bid_differences = stats['bid_differences']
        Q_differences = stats['Q_differences']
        assert sorted(bid_differences.keys()) == sorted(Q_differences.keys())

        plot_3d_totem_scatter(
            data=bid_differences, 
            labels={'x': 'State', 'y': 'Next State', 'z': 'Bid Difference'}, 
            figname=os.path.join(self.quantitative_dir, 'bid_diff_steps{}.png'.format(self.steps)))

        plot_3d_totem_scatter(
            data=Q_differences, 
            labels={'x': 'State', 'y': 'Next State', 'z': 'Q Difference'}, 
            figname=os.path.join(self.quantitative_dir, 'Q_diff_steps{}.png'.format(self.steps)))

        # flatten everything right now
        all_bid_differences = []
        all_Q_differences = []
        for state in bid_differences.keys():
            all_bid_differences.extend(bid_differences[state])
            all_Q_differences.extend(Q_differences[state])
        plt.scatter(all_bid_differences, all_Q_differences)
        plt.xlabel('bid(t+1) - bid(t)')
        plt.ylabel('Q(t) - Q(t+1)')
        plt.savefig(os.path.join(self.quantitative_dir, 'bid_Q_diff_corr_steps{}.png'.format(self.steps)))
        # plt.close()


    ##################################################



class VisualEnvManager(EnvManager):
    def __init__(self, env_name, env_registry, args):
        super(VisualEnvManager, self).__init__(env_name, env_registry, args)
        self.visual = True

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

    def save_gif(self, prefix, gifname, epoch, test_example, remove_images):
        def get_key(fname):
            basename = os.path.basename(fname)
            delimiter = '_t'
            start = basename.rfind(delimiter)
            key = int(basename[start+len(delimiter):-len('.png')])
            return key
        fnames = sorted(glob.glob('{}/{}_e{}_n{}_t*.png'.format(self.qualitative_dir, prefix, epoch, test_example)), key=get_key)
        images = [imageio.imread(fname) for fname in fnames]
        imshape = images[0].shape
        for pad in range(2):
            images.append(imageio.core.util.Array(np.ones(imshape).astype(np.uint8)*255))
        imageio.mimsave(os.path.join(self.qualitative_dir, gifname), images)
        if remove_images:
            for fname in fnames:
                os.remove(fname)

    def save_video(self, epoch, test_example, bids, ret, frames):
        # for i, frame in tqdm(enumerate(frames)):
        for i, frame in enumerate(frames):
            fname = '{}_e{}_n{}_t{}.png'.format(self.env_name, epoch, test_example, i)
            agent_ids = sorted(bids.keys())
            bids_t = [(agent_id, bids[agent_id][i]) for agent_id in agent_ids]
            self.save_image(fname, i, ret, frame, bids_t)
        gifname = 'vid{}_{}_{}.gif'.format(self.env_name, epoch, test_example)
        self.save_gif(self.env_name, gifname, epoch, test_example, remove_images=True)

class GymEnvManager(VisualEnvManager):
    def __init__(self, env_name, env_registry, args):
        super(GymEnvManager, self).__init__(env_name, env_registry, args)
        self.state_dim = self.env.observation_space.shape[0]
        self.is_disc_action = len(self.env.action_space.shape) == 0
        self.action_dim = self.env.action_space.n if self.is_disc_action else self.env.action_space.shape[0]
        self.max_episode_length = self.env._max_episode_steps

class MinigridEnvManager(VisualEnvManager):
    def __init__(self, env_name, env_registry, args):
        super(MinigridEnvManager, self).__init__(env_name, env_registry, args)
        full_state_dim = self.env.observation_space.shape  # (H, W, C)
        self.state_dim = full_state_dim[:-1]  # (H, W)
        self.is_disc_action = len(self.env.action_space.shape) == 0
        self.action_dim = self.env.action_space.n if self.is_disc_action else self.env.action_space.shape[0]
        self.max_episode_length = self.env.max_steps

def log_string(ordered_dict):
    s = ''
    for i, (k, v) in enumerate(ordered_dict.items()):
        delim = '' if i == 0 else ' | '
        if is_float(v):
            s += delim + '{}: {:.5f}'.format(k, v)
        else:
            s += delim + '{}: {}'.format(k, v)
    return s

def format_log_string(list_of_rows):
    length = max(len(s) for s in list_of_rows)
    outside_border = '#'*length
    inside_border = '*'*length
    s = '\n'.join([outside_border]+list_of_rows+[outside_border])
    return s
