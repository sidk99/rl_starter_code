import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
import ujson

from starter_code.utils import all_same, get_first_key
from starter_code.log import mkdirp

import pprint

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
    def __init__(self, exp_subroot):
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
        for mode_dir in [x for x in os.listdir(os.path.join(fp_seed_dir, 'group_0')) if 'train' in x or 'test' in x]:
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
    def __init__(self, exp_subroot):
        Plotter.__init__(self, exp_subroot)

    def load_stats_for_mode(self, mode, fp_mode_dir):
        mode_dict = super(MultiAgentPlotter, self).load_stats_for_mode(mode, fp_mode_dir)
        # load agent stats
        agent_stats_file = os.path.join(fp_mode_dir, 'quantitative', 'agent_state_stats.json')
        mode_dict['agent'] = ujson.load(open(agent_stats_file, 'r'))  
        # note that all keys are strings
        return mode_dict

class CurvePlotter(Plotter):
    def __init__(self, exp_subroot, quantile=True):
        Plotter.__init__(self, exp_subroot)
        self.quantile = quantile

    def align_x_and_y(self, xs, ys):
        if all_same(xs):
            run_x = xs[0]  # since they are the same just take the first one
        else:
            run_x = []
            idx = 0
            while idx < min(len(run_x) for run_x in xs):
                step = xs[0][idx]
                xs_for_this_step = [_run_x[idx] for _run_x in xs]
                if all_same(xs_for_this_step):
                    run_x.append(step)
                    idx += 1
                else:
                    # find the minimum. Delete all the minimums
                    step_with_min_value = np.min(xs_for_this_step)

                    print('Inconsistent row: {}'.format(xs_for_this_step))
                    # if I delete I will mutate!
                    for seed_idx, seed_value in enumerate(xs_for_this_step):
                        if seed_value == step_with_min_value:
                            # mutates!
                            print('Deleting x: {} y: {}'.format(
                                xs[seed_idx][idx], ys[seed_idx][idx]))
                            del xs[seed_idx][idx]  
                            del ys[seed_idx][idx]
                    # note that we do not advance the idx

        # At this point, everything up to len(run_x) is aligned
        ys = [run_y[:len(run_x)] for run_y in ys]
        ys = np.stack(ys)
        assert ys.shape[-1] == len(run_x)
        return run_x, ys

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

    def calculate_error_bars(self, ys):
        if not self.quantile:
            centers = np.mean(ys, axis=0)
            stds = np.std(ys, axis=0)
            mins, maxs = centers-stds, centers+stds
        else:
            centers = np.median(ys, axis=0)
            mins = np.percentile(ys, 10, axis=0)
            maxs = np.percentile(ys, 90, axis=0)
        return centers, mins, maxs

    def fill_plot(self, run_x, ys, **kwargs):
        centers, mins, maxs = self.calculate_error_bars(ys)
        self.custom_plot(run_x, centers, mins, maxs, **kwargs) 

    def plot(self, fname, curve_plot_dict, metric, x_label='steps', title=None):
        # get color
        colors = plt.cm.viridis(np.linspace(0,1,len(curve_plot_dict)))

        # plot data
        for i, label in enumerate(curve_plot_dict):
            self.fill_plot(curve_plot_dict[label]['x'], curve_plot_dict[label]['ys'], 
                label=label, color=colors[i])

        # axes labels and legend
        self.apply_labels(x_label, metric)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        # title
        if title: plt.title(title)

        # save in exp_subroot
        plt.savefig(os.path.join(EXP_ROOT, self.exp_subroot, fname), bbox_inches="tight")
        plt.close()

    def reorganize_episode_data(self, stats_dict, mode, metric, x_label):
        exp_x_y = dict()
        for label in stats_dict:
            xs = []
            ys = []
            for seed in stats_dict[label]:
                xs.append(stats_dict[label][seed][mode]['global'][x_label].tolist())
                ys.append(stats_dict[label][seed][mode]['global'][metric].tolist())
            run_x, ys = self.align_x_and_y(xs, ys)

            exp_x_y[label] = dict(x=run_x, ys=ys)
        return exp_x_y

    def plot_episode_metrics(self, fname, stats_dict, mode, metric, x_label='steps', title=None):
        """
            top-level keys in stats_dict are the labels
        """
        print('Plotting {} for metric {} mode {}'.format(fname, metric, mode))
        # get data
        curve_plot_dict = self.reorganize_episode_data(stats_dict, mode, metric, x_label)
        self.plot('{}_{}_{}.png'.format(fname, metric, mode), curve_plot_dict, metric, x_label, title)


class MultiAgentCurvePlotter(MultiAgentPlotter, CurvePlotter):
    def __init__(self, exp_subroot, quantile=True):
        CurvePlotter.__init__(self, exp_subroot, quantile)

    def reorganize_state_data(self, stats_dict, mode, metric, x_label):
        """
        return 
            d[state][a_id][x]
            d[state][a_id][ys]
        """
        assert len(stats_dict) == 1
        stats_dict = get_first_key(stats_dict)

        reorganized_dict = dict()  

        for seed in stats_dict:
            for state in stats_dict[seed][mode]['agent'][metric]:
                subdict = stats_dict[seed][mode]['agent'][metric][state]
                reorganized_dict[state] = {a_id: dict() for a_id in subdict}

        for state in reorganized_dict:
            for a_id in reorganized_dict[state]:
                xs = []
                ys = []
                for seed in stats_dict:
                    subdict = stats_dict[seed][mode]['agent'][metric][state]
                    run_x = []
                    run_y = []
                    for step in subdict[a_id]:
                        run_x.append(int(step))
                        run_y.append(subdict[a_id][step])
                    xs.append(run_x)
                    ys.append(run_y)

                run_x, ys = self.align_x_and_y(xs, ys)
                reorganized_dict[state][a_id]['x'] = run_x
                reorganized_dict[state][a_id]['ys'] = ys

        return reorganized_dict

    # actually to be honest I don't think it makes sense to compare multiple experiments here
    def plot_state_metrics(self, fname, stats_dict, mode, metric, x_label='steps', title=None):
        print('Plotting {} for metric {} mode {}'.format(fname, metric, mode))
        assert 'bid' in metric or 'payoff' in metric

        # get data
        reorganized_data = self.reorganize_state_data(stats_dict, mode, metric, x_label)

        # want to do it for each state
        for state in reorganized_data:
            curve_plot_dict = reorganized_data[state]
            self.plot('{}_state_{}_{}_{}.png'.format(fname, state, metric, mode), curve_plot_dict, metric, x_label, title)

###############################################################################################

def plot_1_1_20_debug_return():
    p = CurvePlotter(exp_subroot='debug_plot')
    stats_dict = p.load_all_stats(exp_dirs={
        'English': 'CP-0_plr4e-05_optadam_ppo_aucbb_red2_ec0',
        'Vickrey': 'CP-0_plr4e-05_optadam_ppo_aucv_red2_ec0'})
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='English_vs_Vickrey', stats_dict=stats_dict, mode=mode, metric=metric)

def plot_1_1_20_debug_state_metrics():
    p = MultiAgentCurvePlotter(exp_subroot='debug')
    stats_dict = p.load_all_stats(exp_dirs={
        'Redundancy 4': '1S1T1A_plr4e-05_optsgd_ppo_aucv_red4_ec0'})
    p.plot_state_metrics(fname='Redundancy_4', stats_dict=stats_dict, mode='train', metric='mean_payoff')

def plot_1_1_20_debug_return_hdim32():
    p = CurvePlotter(exp_subroot='debug_plot_hdim32')
    stats_dict = p.load_all_stats(exp_dirs={
        'English': 'CP-0_plr4e-05_optadam_ppo_aucbb_red2_ec0',
        'Vickrey': 'CP-0_plr4e-05_optadam_ppo_aucv_red2_ec0'})
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='English_vs_Vickrey', stats_dict=stats_dict, mode=mode, metric=metric)

def plot_1_2_20_debug_return_hdim16():
    p = CurvePlotter(exp_subroot='debug_plot_hdim16')
    stats_dict = p.load_all_stats(exp_dirs={
        'English': 'CP-0_plr4e-05_optadam_ppo_aucbb_red2_ec0',
        'Vickrey': 'CP-0_plr4e-05_optadam_ppo_aucv_red2_ec0'})
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='English_vs_Vickrey', stats_dict=stats_dict, mode=mode, metric=metric)

def plot_vickrey_chain_debug_geb():
    p = MultiAgentCurvePlotter(exp_subroot='server/debug_vickrey_chain_geb')

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_2' : 'CW2_plr4e-05_optadam_ppo_aucv_red2_ec0',
        })
    p.plot_state_metrics(fname='Chain_2', stats_dict=stats_dict, mode='train', metric='mean_payoff')
    p.plot_state_metrics(fname='Chain_2', stats_dict=stats_dict, mode='train', metric='mean_bid')

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_3' : 'CW3_plr4e-05_optadam_ppo_aucv_red2_ec0',
        })
    p.plot_state_metrics(fname='Chain_3', stats_dict=stats_dict, mode='train', metric='mean_payoff')
    p.plot_state_metrics(fname='Chain_3', stats_dict=stats_dict, mode='train', metric='mean_bid')

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_4' : 'CW4_plr4e-05_optadam_ppo_aucv_red2_ec0',
        })
    p.plot_state_metrics(fname='Chain_4', stats_dict=stats_dict, mode='train', metric='mean_payoff')
    p.plot_state_metrics(fname='Chain_4', stats_dict=stats_dict, mode='train', metric='mean_bid')

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_5' : 'CW5_plr4e-05_optadam_ppo_aucv_red2_ec0',
        })
    p.plot_state_metrics(fname='Chain_5', stats_dict=stats_dict, mode='train', metric='mean_payoff')
    p.plot_state_metrics(fname='Chain_5', stats_dict=stats_dict, mode='train', metric='mean_bid')

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_6' : 'CW6_plr4e-05_optadam_ppo_aucv_red2_ec0',
        })
    p.plot_state_metrics(fname='Chain_6', stats_dict=stats_dict, mode='train', metric='mean_payoff')
    p.plot_state_metrics(fname='Chain_6', stats_dict=stats_dict, mode='train', metric='mean_bid')

def plot_bandit_geb():
    pass




if __name__ == '__main__':
    # plot_vickrey_chain_debug_geb()
    # plot_1_1_20_debug_return()
    # plot_1_2_20_debug_return_hdim16()
    plot_1_1_20_debug_return_hdim32()




