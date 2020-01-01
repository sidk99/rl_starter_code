import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
import ujson

from starter_code.utils import all_same
from starter_code.log import mkdirp

"""
subroot
    - exp_0 (expdir)
        - seed_0_<datetime> (logdir)
            - code
            - group_0
                - <env_name>_test
                    - quantitative
                        - agent_state_stats.json
                        - exp_name.csv
                - <env_name>_train
                    - quantitative
                        - agent_state_stats.json
                        - exp_name.csv
        - seed_1_<datetime>
            - etc
    - exp_1
        - etc

What you want is to be able to dump all of these things into the csv/json and then slice it later however you want.

Average over seeds, compare across runs.
"""
PLOT_ROOT = '/Users/michaelchang/Documents/Researchlink/Berkeley/auction_project/plots'
EXP_ROOT = '/Users/michaelchang/Documents/Researchlink/Berkeley/auction_project/runs'

class Plotter():
    def __init__(self, plot_dir, exp_subroot):
        self.plot_dir = os.path.join(PLOT_ROOT, plot_dir)
        self.exp_subroot = exp_subroot

    def load_stats_for_mode(self, mode, fp_mode_dir):
        mode_dict = dict()
        # load global stats
        global_stats_file = os.path.join(fp_mode_dir, 'quantitative', 'global_stats.csv')
        df = pd.read_csv(global_stats_file)
        mode_dict['global'] = df
        return mode_dict

    def load_stats_for_seed(self, seed, fp_seed_dir):
        seed_dict = dict()
        # load params
        seed_dict['params'] = ujson.load(
            open(os.path.join(fp_seed_dir, 'code', 'params.json'), 'r'))
        for mode_dir in [x for x in os.listdir(os.path.join(fp_seed_dir, 'group_0')) if 'train' in x]:# or 'test' in x]:
            mode = mode_dir[mode_dir.rfind('_')+1:]
            fp_mode_dir = os.path.join(fp_seed_dir, 'group_0', mode_dir)
            seed_dict[mode] = self.load_stats_for_mode(mode, fp_mode_dir)
        return seed_dict

    def load_all_stats(self, exp_dirs):
        stats_dict = dict()
        for label, exp_dir in exp_dirs.items():
            stats_dict[label] = dict()
            # now traverse the seeds
            fp_exp_dir = os.path.join(EXP_ROOT, self.exp_subroot, exp_dir)
            for seed_dir in [x for x in os.listdir(fp_exp_dir) if 'seed' in x]:
                fp_seed_dir = os.path.join(fp_exp_dir, seed_dir)
                seed = seed_dir[len('seed'):seed_dir.find('__')]
                stats_dict[label][seed] = self.load_stats_for_seed(seed, fp_seed_dir)
        return stats_dict

class MultiAgentPlotter(Plotter):
    def __init__(self, plot_dir, exp_subroot):
        Plotter.__init__(self, plot_dir, exp_subroot)

    def load_stats_for_mode(self, mode, fp_mode_dir):
        mode_dict = super(MultiAgentPlotter, self).load_stats_for_mode(mode, fp_mode_dir)
        # load agent stats
        agent_stats_file = os.path.join(fp_mode_dir, 'quantitative', 'agent_state_stats.json')
        mode_dict['agent'] = ujson.load(open(agent_stats_file, 'r'))
        return mode_dict

class CurvePlotter(Plotter):
    def __init__(self, plot_dir, exp_subroot):
        Plotter.__init__(self, plot_dir, exp_subroot)

    def get_x_y_for_exp(self, exp_dict, x_label, metric):
        xs = []
        ys = []
        for seed in exp_dict:
            xs.append(exp_dict[seed]['train']['global'][x_label].tolist())
            ys.append(exp_dict[seed]['train']['global'][metric].tolist())

        min_length = min(len(run_x) for run_x in xs)
        xs = [run_x[:min_length] for run_x in xs]
        assert all_same(xs)
        run_x = xs[0]  # since they are the same just take the first one
        ys = np.stack([run_y[:min_length] for run_y in ys])
        return run_x, ys

    def get_x_y(self, stats_dict, x_label, metric):
        exp_x_y = dict()
        for label in stats_dict:
            x, ys= self.get_x_y_for_exp(stats_dict[label], x_label, metric)
            exp_x_y[label] = dict(x=x, ys=ys)
        return exp_x_y

    def apply_labels(self, xlabel, ylabel, outside=True, legend='upper left'):
        if len(legend) > 0:
            if outside:
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                plt.legend(loc=legend)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def custom_plot(self, x, means, mins, maxs, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        base_line, = ax.plot(x, means, **kwargs)
        ax.fill_between(x, mins, maxs, 
            facecolor=base_line.get_color(), alpha=0.5, linewidth=0.0)

    def fill_plot(self, run_x, ys, **kwargs):
        quantile=True

        if not quantile:
            centers = np.mean(ys, axis=0)
            stds = np.std(ys, axis=0)
            mins, maxs = centers-stds, centers+stds
        else:
            centers = np.median(ys, axis=0)
            mins = np.percentile(ys, 10, axis=0)
            maxs = np.percentile(ys, 90, axis=0)

        self.custom_plot(run_x, centers, mins, maxs, **kwargs) 

    def plot(self, fname, stats_dict, metric, x_label='steps', title=None):
        """
            top-level keys in stats_dict are the labels
        """
        # get data
        exp_x_y = self.get_x_y(stats_dict, x_label, metric)

        # get color
        colors = plt.cm.viridis(np.linspace(0,1,len(exp_x_y)))

        # plot data
        for i, label in enumerate(exp_x_y):
            self.fill_plot(exp_x_y[label]['x'], exp_x_y[label]['ys'], 
                label=label, color=colors[i])

        # axes labels and legend
        self.apply_labels(x_label, metric)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        # title
        if title: plt.title(title)

        # save
        plt.savefig(os.path.join(self.plot_dir, '{}_{}.png'.format(fname, metric)), 
            bbox_inches="tight")
        plt.close()

if __name__ == '__main__':
    p = CurvePlotter(plot_dir='debug', exp_subroot='debug_plot')
    stats_dict = p.load_all_stats(exp_dirs={
        'English': 'CP-0_plr4e-05_optadam_ppo_aucbb_red2_ec0',
        'Vickrey': 'CP-0_plr4e-05_optadam_ppo_aucv_red2_ec0'})
    p.plot(fname='English_vs_Vickrey', stats_dict=stats_dict, metric='mean_return')


