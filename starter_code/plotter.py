import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
import shutil
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
                ########
                # killed seed 3 and 4 for minigrid and babyai
                ########
                if 'BAI' in exp_dir or 'MG-E-R' in exp_dir:
                    if 'seed3' in seed_dir or 'seed4' in seed_dir and 'BAI' in exp_dir:
                        continue

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

    def plot(self, fname, curve_plot_dict, metric, x_label='steps', title=None, ylim=None):
        # get color
        colors = plt.cm.viridis(np.linspace(0,1,len(curve_plot_dict)))

        # plot data
        for i, label in sorted(enumerate(curve_plot_dict)):
            self.fill_plot(curve_plot_dict[label]['x'], curve_plot_dict[label]['ys'], 
                label=label, color=colors[i])

        # axes labels and legend
        self.apply_labels(x_label, metric)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        # title
        if not title:
            title = fname
            plt.title(title)

        # ylim
        if ylim:
            plt.ylim(ylim)

        # plt.xscale('log')

        # save in exp_subroot
        plt.savefig(os.path.join(EXP_ROOT, self.exp_subroot, '{}.png'.format(
            fname.replace('.', '-'))), bbox_inches="tight")
        plt.close()

    def unnormalize_returns(self, returns, reward_shift=0, reward_scale=1):
        unnormalized = returns/float(reward_scale) + reward_shift
        return unnormalized

    def handle_unnormalize_returns(self, seed_run_y, metric, params):
        if 'return' in metric:
            reward_params = {}
            for key in params:
                if key in ['reward_scale', 'reward_shift']:
                    reward_params[key] = params[key]
                # if not, will default to reward_scale=1 and reward_shift=0
            seed_run_y = self.unnormalize_returns(seed_run_y, **reward_params)
        return seed_run_y

    def reorganize_episode_data(self, stats_dict, mode, metric, x_label):
        exp_x_y = dict()
        for label in stats_dict:
            xs = []
            ys = []

            for seed in stats_dict[label]:
                seed_run_x = stats_dict[label][seed][mode]['global'][x_label].tolist()
                seed_run_y = np.array(stats_dict[label][seed][mode]['global'][metric].tolist())
                seed_run_y = self.handle_unnormalize_returns(
                    seed_run_y, metric, stats_dict[label][seed]['params'])
                xs.append(seed_run_x)
                ys.append(seed_run_y)
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
        self.plot('{}_{}_{}'.format(fname, metric, mode), curve_plot_dict, metric, x_label, title)


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
            if 'bid' in metric:
                ylim=[0,1]
            elif 'payoff' in metric:
                ylim = [-1, 1]
            else:
                assert False
            self.plot('{}_state_{}_{}_{}'.format(fname, state, metric, mode), curve_plot_dict, metric, x_label, title, ylim=ylim)

    def plot_all_state_metrics(self, fname, stats_dict, mode, metrics, x_label='steps', title=None):
        for metric in metrics:
            self.plot_state_metrics(fname, stats_dict, mode, metric, x_label, title)

    def load_plot_all_state_metrics(self, fname, exp_dir, metrics, mode='test'):
        stats_dict = self.load_all_stats(exp_dirs={fname : exp_dir})
        self.plot_all_state_metrics(fname=fname, stats_dict=stats_dict, mode=mode, metrics=metrics)


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
        'Vickrey': 'CP-0_plr4e-05_optadam_ppo_aucv_red2_ec0',
        'Centralized': 'CP-0_plr4e-05_optadam_ppo_ec0'})
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(
                fname='English_vs_Vickrey_vs_Centralized', 
                stats_dict=stats_dict, mode=mode, metric=metric)

def plot_vickrey_chain_debug_geb():
    p = MultiAgentCurvePlotter(exp_subroot='server/debug_vickrey_chain_geb')

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_2' : 'CW2_plr4e-05_optadam_ppo_aucv_red2_ec0',
        })
    p.plot_all_state_metrics(fname='Chain_2', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_3' : 'CW3_plr4e-05_optadam_ppo_aucv_red2_ec0',
        })
    p.plot_all_state_metrics(fname='Chain_3', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_4' : 'CW4_plr4e-05_optadam_ppo_aucv_red2_ec0',
        })
    p.plot_all_state_metrics(fname='Chain_4', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_5' : 'CW5_plr4e-05_optadam_ppo_aucv_red2_ec0',
        })
    p.plot_all_state_metrics(fname='Chain_5', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_6' : 'CW6_plr4e-05_optadam_ppo_aucv_red2_ec0',
        })
    p.plot_all_state_metrics(fname='Chain_6', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])

def plot_bandit_claude():
    p = MultiAgentCurvePlotter(exp_subroot='server/debug_bandit_claude')


    p.load_plot_all_state_metrics(
        fname='English_2_Arm_Bandit_ec0.0', 
        exp_dir='1S1T2A_plr4e-05_optadam_ppo_aucbb_red2_ec0.0', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='English_2_Arm_Bandit_ec0.001', 
        exp_dir='1S1T2A_plr4e-05_optadam_ppo_aucbb_red2_ec0.001', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='English_2_Arm_Bandit_ec0.01', 
        exp_dir='1S1T2A_plr4e-05_optadam_ppo_aucbb_red2_ec0.01', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='English_2_Arm_Bandit_ec0.1', 
        exp_dir='1S1T2A_plr4e-05_optadam_ppo_aucbb_red2_ec0.1', 
        metrics=['mean_payoff', 'mean_bid'])

    p.load_plot_all_state_metrics(
        fname='English_3_Arm_Bandit_ec0.0', 
        exp_dir='1S1T3A_plr4e-05_optadam_ppo_aucbb_red2_ec0.0', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='English_3_Arm_Bandit_ec0.001', 
        exp_dir='1S1T3A_plr4e-05_optadam_ppo_aucbb_red2_ec0.001', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='English_3_Arm_Bandit_ec0.01', 
        exp_dir='1S1T3A_plr4e-05_optadam_ppo_aucbb_red2_ec0.01', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='English_3_Arm_Bandit_ec0.1', 
        exp_dir='1S1T3A_plr4e-05_optadam_ppo_aucbb_red2_ec0.1', 
        metrics=['mean_payoff', 'mean_bid'])

    p.load_plot_all_state_metrics(
        fname='English_4_Arm_Bandit_ec0.0', 
        exp_dir='1S1T4A_plr4e-05_optadam_ppo_aucbb_red2_ec0.0', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='English_4_Arm_Bandit_ec0.001', 
        exp_dir='1S1T4A_plr4e-05_optadam_ppo_aucbb_red2_ec0.001', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='English_4_Arm_Bandit_ec0.01', 
        exp_dir='1S1T4A_plr4e-05_optadam_ppo_aucbb_red2_ec0.01', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='English_4_Arm_Bandit_ec0.1', 
        exp_dir='1S1T4A_plr4e-05_optadam_ppo_aucbb_red2_ec0.1', 
        metrics=['mean_payoff', 'mean_bid'])

    p.load_plot_all_state_metrics(
        fname='English_9_Arm_Bandit_ec0.0', 
        exp_dir='1S1T9A_plr4e-05_optadam_ppo_aucbb_red2_ec0.0', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='English_9_Arm_Bandit_ec0.001', 
        exp_dir='1S1T9A_plr4e-05_optadam_ppo_aucbb_red2_ec0.001', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='English_9_Arm_Bandit_ec0.01', 
        exp_dir='1S1T9A_plr4e-05_optadam_ppo_aucbb_red2_ec0.01', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='English_9_Arm_Bandit_ec0.1', 
        exp_dir='1S1T9A_plr4e-05_optadam_ppo_aucbb_red2_ec0.1', 
        metrics=['mean_payoff', 'mean_bid'])


    p.load_plot_all_state_metrics(
        fname='Vickrey_2_Arm_Bandit_ec0.0', 
        exp_dir='1S1T2A_plr4e-05_optadam_ppo_aucv_red2_ec0.0', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='Vickrey_2_Arm_Bandit_ec0.001', 
        exp_dir='1S1T2A_plr4e-05_optadam_ppo_aucv_red2_ec0.001', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='Vickrey_2_Arm_Bandit_ec0.01', 
        exp_dir='1S1T2A_plr4e-05_optadam_ppo_aucv_red2_ec0.01', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='Vickrey_2_Arm_Bandit_ec0.1', 
        exp_dir='1S1T2A_plr4e-05_optadam_ppo_aucv_red2_ec0.1', 
        metrics=['mean_payoff', 'mean_bid'])

    p.load_plot_all_state_metrics(
        fname='Vickrey_3_Arm_Bandit_ec0.0', 
        exp_dir='1S1T3A_plr4e-05_optadam_ppo_aucv_red2_ec0.0', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='Vickrey_3_Arm_Bandit_ec0.001', 
        exp_dir='1S1T3A_plr4e-05_optadam_ppo_aucv_red2_ec0.001', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='Vickrey_3_Arm_Bandit_ec0.01', 
        exp_dir='1S1T3A_plr4e-05_optadam_ppo_aucv_red2_ec0.01', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='Vickrey_3_Arm_Bandit_ec0.1', 
        exp_dir='1S1T3A_plr4e-05_optadam_ppo_aucv_red2_ec0.1', 
        metrics=['mean_payoff', 'mean_bid'])

    p.load_plot_all_state_metrics(
        fname='Vickrey_4_Arm_Bandit_ec0.0', 
        exp_dir='1S1T4A_plr4e-05_optadam_ppo_aucv_red2_ec0.0', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='Vickrey_4_Arm_Bandit_ec0.001', 
        exp_dir='1S1T4A_plr4e-05_optadam_ppo_aucv_red2_ec0.001', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='Vickrey_4_Arm_Bandit_ec0.01', 
        exp_dir='1S1T4A_plr4e-05_optadam_ppo_aucv_red2_ec0.01', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='Vickrey_4_Arm_Bandit_ec0.1', 
        exp_dir='1S1T4A_plr4e-05_optadam_ppo_aucv_red2_ec0.1', 
        metrics=['mean_payoff', 'mean_bid'])

    p.load_plot_all_state_metrics(
        fname='Vickrey_9_Arm_Bandit_ec0.0', 
        exp_dir='1S1T9A_plr4e-05_optadam_ppo_aucv_red2_ec0.0', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='Vickrey_9_Arm_Bandit_ec0.001', 
        exp_dir='1S1T9A_plr4e-05_optadam_ppo_aucv_red2_ec0.001', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='Vickrey_9_Arm_Bandit_ec0.01', 
        exp_dir='1S1T9A_plr4e-05_optadam_ppo_aucv_red2_ec0.01', 
        metrics=['mean_payoff', 'mean_bid'])
    p.load_plot_all_state_metrics(
        fname='Vickrey_9_Arm_Bandit_ec0.1', 
        exp_dir='1S1T9A_plr4e-05_optadam_ppo_aucv_red2_ec0.1', 
        metrics=['mean_payoff', 'mean_bid'])



def plot_english_chain_debug_geb():
    p = MultiAgentCurvePlotter(exp_subroot='server/debug_english_chain_geb')

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_2' : 'CW2_plr4e-05_optadam_ppo_aucbb_red2_ec0',
        })
    p.plot_all_state_metrics(fname='Chain_2', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_3' : 'CW3_plr4e-05_optadam_ppo_aucbb_red2_ec0',
        })
    p.plot_all_state_metrics(fname='Chain_3', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_4' : 'CW4_plr4e-05_optadam_ppo_aucbb_red2_ec0',
        })
    p.plot_all_state_metrics(fname='Chain_4', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_5' : 'CW5_plr4e-05_optadam_ppo_aucbb_red2_ec0',
        })
    p.plot_all_state_metrics(fname='Chain_5', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_6' : 'CW6_plr4e-05_optadam_ppo_aucbb_red2_ec0',
        })
    p.plot_all_state_metrics(fname='Chain_6', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])


def plot_entropy_chain_debug_claude():
    p = MultiAgentCurvePlotter(exp_subroot='server/debug_chain_claude_entropy')

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_2_English_ec0.001' : 'CW2_plr4e-05_optadam_ppo_aucbb_red2_ec0.001',
        })
    p.plot_all_state_metrics(fname='Chain_2_English_ec0.001', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_2_Vickrey_ec0.001' : 'CW2_plr4e-05_optadam_ppo_aucv_red2_ec0.001',
        })
    p.plot_all_state_metrics(fname='Chain_2_Vickrey_ec0.001', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_4_English_ec0.001' : 'CW4_plr4e-05_optadam_ppo_aucbb_red2_ec0.001',
        })
    p.plot_all_state_metrics(fname='Chain_4_English_ec0.001', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_4_Vickrey_ec0.001' : 'CW4_plr4e-05_optadam_ppo_aucv_red2_ec0.001',
        })
    p.plot_all_state_metrics(fname='Chain_4_Vickrey_ec0.001', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_6_English_ec0.001' : 'CW6_plr4e-05_optadam_ppo_aucbb_red2_ec0.001',
        })
    p.plot_all_state_metrics(fname='Chain_6_English_ec0.001', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])

    stats_dict = p.load_all_stats(exp_dirs={
        'Chain_6_Vickrey_ec0.001' : 'CW6_plr4e-05_optadam_ppo_aucv_red2_ec0.001',
        })
    p.plot_all_state_metrics(fname='Chain_6_Vickrey_ec0.001', stats_dict=stats_dict, mode='train', metrics=['mean_payoff', 'mean_bid'])



def plot_1_6_20_debug_lunarlander():
    p = CurvePlotter(exp_subroot='server/debug_lunarlander_geb')
    stats_dict = p.load_all_stats(exp_dirs={
        'FPBSA_ec0.0': 'LL-2_plr4e-05_optadam_ppo_aucbb_red2_ec0.0',
        'FPBSA_ec0.001': 'LL-2_plr4e-05_optadam_ppo_aucbb_red2_ec0.001',
        'Vickrey_ec0.0': 'LL-2_plr4e-05_optadam_ppo_aucv_red2_ec0.0',
        'Vickrey_ec0.001': 'LL-2_plr4e-05_optadam_ppo_aucv_red2_ec0.001',
        })
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='LunarLander', stats_dict=stats_dict, mode=mode, metric=metric)


def plot_1_7_20_CP_redundancy_1():
    p = CurvePlotter(exp_subroot='debug_redundancy_CP_geb')
    stats_dict = p.load_all_stats(exp_dirs={
        'Vickrey': 'CP-0_plr4e-05_optadam_ppo_aucbb_red1_ec0',
        'FPBSA': 'CP-0_plr4e-05_optadam_ppo_aucv_red1_ec0',
        })
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='CP_Redundancy_1', stats_dict=stats_dict, mode=mode, metric=metric)


def plot_1_13_20_CW6_sparse():
    """ 
        Note that here you get 0 reward for every step, and discount is 1.
        Vickrey can do it if it is not sparse reward.

        In sparse reward scenarios, we observe that FPBSA does better, whereas Vickrey dies with redundancy 1 and with redundancy 2 doesn't converge to optimal solution.

        The discount factor was 0.99 here.
        You need to scale the discount factor based on the horizon length.

        What we observe:
            - FPBSA
                - redundancy 1: bids 0 everywhere and only slightly above 0 at the last step
                    - ec0: if it is truly sparse reward, then FPBSA bids 0 everywhere and only slightly above 0 at the last step, which is as expected. Note that it bids very low. The payoff everywhere is also 0, except at the last timestep; no return decomposition.
                    - ec0.001: same behavior as ec0
                - redundancy 2
                    - ec0: the payoff at every step is about 0.05. So there is return decomposition here. They all bid slightly under the Q-value, but still pretty high up - possibly because of the redundancy.
                    - ec0.001: 
            - Vickrey
                - redundancy 1: 
                    -ec0: There is high variance in the bids. Perhaps this is due the fact that there is a gap between the second highest bid and the first highest bid. I observe by random chance, the worse action (which gets 0 terminal reward) accidentally won and kept on bidding high, whereas the better action kept on getting 0 reward because it kept on losing. So the worse action actually ended up bidding about 0.6. What's interesting is that before that timestep, the best action actually outbid the worse action, and bid exactly 0.6, which is the Q-value. So I think in order to fix this we need them to sufficiently be able to explore. SAC helps with this.
                    - ec0.001: this actually gave the worse action more chances at winning. And since credit is not conserved, the later agents' bid will become a target for previous agents, and then this will mess up the early part of the chain. What we want is that the optimal agent to get a chance to win at the last timestep. If that happens, then by induction we get the result we want.
                - redundancy 2: 
                    - ec0: ok what we observe here in the sparse reward setting is that there is no incentive for the worse actions to not bid high, because if you are truly sparse reward, and there is no time pressure, then you can just go back and forth in the same state. Especially since here the discount factor is close to 1. So without a penalty for taking actions, then you don't get optimal reward all the time, because you just end up losing sometime, and then the episode ends before you can recover. 

        Note also that the episode length here was basically just 4, so if you died before the horizon length hit, then you don't get to see the full state space.
        So I think one confounding factor to this experiment was that the horizon length was too short. 

        The next action item is to redo this experiment, but with a longer horizon length.
        Let's see if PPO exploration is sufficient, and the gamma is sufficient. 
        If not, let's see if SAC works.

        Questions:
            - how does the redundancy affect the FPBSA? Does it make it closer to Vickrey?

    """
    p = MultiAgentCurvePlotter(exp_subroot='server/debug_chain_claude_entropy_sparse')
    exp_dirs = {
        # 'FPBSA_red1_ec0': 'CW6_plr4e-05_optadam_ppo_aucbb_red1_ec0.0',
        # 'FPBSA_red1_ec0-001': 'CW6_plr4e-05_optadam_ppo_aucbb_red1_ec0.001',
        # 'FPBSA_red2_ec0': 'CW6_plr4e-05_optadam_ppo_aucbb_red2_ec0.0',
        # 'FPBSA_red2_ec0-001': 'CW6_plr4e-05_optadam_ppo_aucbb_red2_ec0.001',
        'Vickrey_red1_ec0': 'CW6_plr4e-05_optadam_ppo_aucv_red1_ec0.0',
        'Vickrey_red1_ec0-001': 'CW6_plr4e-05_optadam_ppo_aucv_red1_ec0.001',
        # 'Vickrey_red2_ec0': 'CW6_plr4e-05_optadam_ppo_aucv_red2_ec0.0',
        # 'Vickrey_red2_ec0-001': 'CW6_plr4e-05_optadam_ppo_aucv_red2_ec0.001',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='Chain_Sparse_Redundancy', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    for fname, exp_dir in exp_dirs.items():
        p.load_plot_all_state_metrics(
            fname=fname, 
            exp_dir=exp_dir, 
            metrics=['mean_payoff', 'mean_bid'])




def plot_1_13_20_debug_lunarlander_h():
    """
        Observations: 
            16-16 and 32-32 are less stable to train.
            64 Vickrey also doesn't that great
            Overall though it's not getting great reward.
            Also the bids for everyone is basically 0.5. Not sure why.

    """
    p = CurvePlotter(exp_subroot='server/debug_lunarlander_geb_h')
    stats_dict = p.load_all_stats(exp_dirs={
        # 'FPBSA_16-16': 'LL-2_g0.99_plr4e-05_ppo_h16-16_aucbb_red2_ec0',
        # 'Vickrey_16-16': 'LL-2_g0.99_plr4e-05_ppo_h16-16_aucv_red2_ec0',
        # 'FPBSA_32-32': 'LL-2_g0.99_plr4e-05_ppo_h32-32_aucbb_red2_ec0',
        # 'Vickrey_32-32': 'LL-2_g0.99_plr4e-05_ppo_h32-32_aucv_red2_ec0',
        'FPBSA_32': 'LL-2_g0.99_plr4e-05_ppo_h32_aucbb_red2_ec0',
        'Vickrey_32': 'LL-2_g0.99_plr4e-05_ppo_h32_aucv_red2_ec0',
        'FPBSA_64-64': 'LL-2_g0.99_plr4e-05_ppo_h64-64_aucbb_red2_ec0',
        'Vickrey_64-64': 'LL-2_g0.99_plr4e-05_ppo_h64-64_aucv_red2_ec0',
        'FPBSA_64': 'LL-2_g0.99_plr4e-05_ppo_h64_aucbb_red2_ec0',
        # 'Vickrey_64': 'LL-2_g0.99_plr4e-05_ppo_h64_aucv_red2_ec0',
        })
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='LunarLander', stats_dict=stats_dict, mode=mode, metric=metric)


def plot_1_20_20_debug_reward_scaling():
    """ 
    """
    p = CurvePlotter(exp_subroot='debug')
    exp_dirs = {
        'CartPole': 'CP-0_g0.99_plr4e-05_ppo_h20-20_db_aucv_red2_ec0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='debug_reward_scaling', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    # for fname, exp_dir in exp_dirs.items():
    #     p.load_plot_all_state_metrics(
    #         fname=fname, 
    #         exp_dir=exp_dir, 
    #         metrics=['mean_payoff', 'mean_bid'])



def plot_1_19_20_debug_minigrid_geb_h():
    """
        Observations: 
            bucket brigrade seems to be better than vickrey in practice
            perhaps this is because vickrey has an unbounded price of anarchy
            so one potential solution is to try better exploration, like SAC.

        Next steps
            - Try better exploration
    """
    p = CurvePlotter(exp_subroot='server/debug_minigrid_geb_h')
    stats_dict = p.load_all_stats(exp_dirs={
        'Bucket Brigade ec0': 'BAI-OOD-0_g0.99_plr4e-05_ppo_h16_aucbb_red2_ec0.0',
        'Bucket Brigade ec0.001': 'BAI-OOD-0_g0.99_plr4e-05_ppo_h16_aucbb_red2_ec0.001',
        'Bucket Brigade ec0.01': 'BAI-OOD-0_g0.99_plr4e-05_ppo_h16_aucbb_red2_ec0.01',
        'Vickrey ec0': 'BAI-OOD-0_g0.99_plr4e-05_ppo_h16_aucv_red2_ec0.0',
        'Vickrey ec0.001': 'BAI-OOD-0_g0.99_plr4e-05_ppo_h16_aucv_red2_ec0.001',
        'Vickrey ec0.01': 'BAI-OOD-0_g0.99_plr4e-05_ppo_h16_aucv_red2_ec0.01',
        })
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='MiniGrid Open-One-Door', stats_dict=stats_dict, mode=mode, metric=metric)

    p = CurvePlotter(exp_subroot='server/debug_minigrid_geb_h')
    stats_dict = p.load_all_stats(exp_dirs={
        'Bucket Brigade ec0': 'BAI-PK-0_g0.99_plr4e-05_ppo_h16_aucbb_red2_ec0.0',
        'Bucket Brigade ec0.001': 'BAI-PK-0_g0.99_plr4e-05_ppo_h16_aucbb_red2_ec0.001',
        'Bucket Brigade ec0.01': 'BAI-PK-0_g0.99_plr4e-05_ppo_h16_aucbb_red2_ec0.01',
        'Vickrey ec0': 'BAI-PK-0_g0.99_plr4e-05_ppo_h16_aucv_red2_ec0.0',
        'Vickrey ec0.001': 'BAI-PK-0_g0.99_plr4e-05_ppo_h16_aucv_red2_ec0.001',
        'Vickrey ec0.01': 'BAI-PK-0_g0.99_plr4e-05_ppo_h16_aucv_red2_ec0.01',
        })
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='MiniGrid Pickup-Key', stats_dict=stats_dict, mode=mode, metric=metric)



def plot_1_27_20_bandit_ado():
    """ 
        Redundancy 2 gets higher performance
        Vickrey converges faster than English

    """
    p = MultiAgentCurvePlotter(exp_subroot='server/bandit_ado')
    exp_dirs = {
        # we have results here, but not necessary for the plot
        # 'Bucket_Brigade_Redundancy_1_ec0-001': '1S1T4A_g0.99_plr4e-05_ppo_h16_aucbb_ado_red1_ec0.001',
        # 'Bucket_Brigade_Redundancy_2_ec0-001': '1S1T4A_g0.99_plr4e-05_ppo_h16_aucbb_ado_red2_ec0.001',
        # 'Vickrey_Redundancy_1_ec0-001': '1S1T4A_g0.99_plr4e-05_ppo_h16_aucv_ado_red1_ec0.001',
        # 'Vickrey_Redundancy_2_ec0-001': '1S1T4A_g0.99_plr4e-05_ppo_h16_aucv_ado_red2_ec0.001',



        'Bucket_Brigade_Redundancy_1_ec0': '1S1T4A_g0.99_plr4e-05_ppo_h16_aucbb_ado_red1_ec0.0',
        'Bucket_Brigade_Redundancy_2_ec0': '1S1T4A_g0.99_plr4e-05_ppo_h16_aucbb_ado_red2_ec0.0',
        'Bucket_Brigade_clone_ec0': '1S1T4A_g0.99_plr4e-05_ppo_h16_cln_aucbb_ado_red2_ec0.0',

        
        'Vickrey_Redundancy_1_ec0': '1S1T4A_g0.99_plr4e-05_ppo_h16_aucv_ado_red1_ec0.0',
        'Vickrey_Redundancy_2_ec0': '1S1T4A_g0.99_plr4e-05_ppo_h16_aucv_ado_red2_ec0.0',
        'Vickrey_clone_ec0': '1S1T4A_g0.99_plr4e-05_ppo_h16_cln_aucv_ado_red2_ec0.0',


        'Credit_Conserving_Vickrey_Redundancy_1_ec0': '1S1T4A_g0.99_plr4e-05_ppo_h16_aucccv_ado_red1_ec0.0',
        'Credit_Conserving_Vickrey_Redundancy_2_ec0': '1S1T4A_g0.99_plr4e-05_ppo_h16_aucccv_ado_red2_ec0.0',
        'Credit_Conserving_Vickrey_clone_ec0': '1S1T4A_g0.99_plr4e-05_ppo_h16_cln_aucccv_ado_red2_ec0.0',



        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='Bandit Agent Dropout', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    for fname, exp_dir in exp_dirs.items():
        p.load_plot_all_state_metrics(
            fname=fname, 
            exp_dir=exp_dir, 
            metrics=['mean_payoff', 'mean_bid'])

            
def plot_1_27_20_debug_atari_breakout():
    """
        Purpose: trying to figure out the reward range is for horizon 100
    """
    p = CurvePlotter(exp_subroot='debug')
    exp_dirs = {
        'Breakout_ec0': 'B--4_g0.99_plr4e-05_ppo_h128-128_ec0',
        'Breakout_ec0-01': 'B--4_g0.99_plr4e-05_ppo_h128-128_ec0.01',
        'Breakout_ec1': 'B--4_g0.99_plr4e-05_ppo_h128-128_ec1.0',
        'Breakout_nfs_ec0': 'B-NF-4_g0.99_plr4e-05_ppo_h128-128_ec0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='debug_atari_breakout', 
                stats_dict=stats_dict, mode=mode, metric=metric)


def plot_1_28_20_debug_babyai_sac():
    """
        Purpose: trying to figure out the reward range is for horizon 100
    """
    p = CurvePlotter(exp_subroot='server/debug_minigrid_geb_sac')
    exp_dirs = {
            'BucketBrigade_PickupKey': 'BAI-PK-0_g0.99_plr4e-05_sac_h16_aucbb_red2_ec0',
            'Vickrey_PickupKey': 'BAI-PK-0_g0.99_plr4e-05_sac_h16_aucv_red2_ec0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='debug_minigrid_pickupkey_geb_sac', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    exp_dirs = {
            'BucketBrigade_OpenOneDoor': 'BAI-OOD-0_g0.99_plr4e-05_sac_h16_aucbb_red2_ec0',
            'Vickrey_OpenOneDoor': 'BAI-OOD-0_g0.99_plr4e-05_sac_h16_aucv_red2_ec0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='debug_minigrid_openonedoor_geb_sac', 
                stats_dict=stats_dict, mode=mode, metric=metric)



def plot_1_28_20_CW6_long_horizon():
    """ 
        The hope for this experiment was to use longer horizon to allow the agent to hit the goal before being cut off
        Note that this is a sparse reward

        Global Objective
            BucketBrigade 
                gamma 0.99: solves it
                gamma 1: only fails when redundancy is 1 and ec is 0

            Vickrey
                gamma 0.99: fails
                gamma 1: fails

        the exploration bonus either is too low or not having an effect

        Now let's look at what the bidding behavior is
            BucketBrigade
                gamma 0.99
                    redundancy 1
                        ec 0: winner bids very close to 0, but just slightly; gets payoff of 0 at all steps except the last
                        ec 0.001: 
                    redundancy 2
                        ec 0: 
                        ec 0.001

                gamma 1
                    redundancy 1
                        ec 0
                        ec 0.001
                    redundancy 2
                        ec 0
                        ec 0.001

    """
    p = MultiAgentCurvePlotter(exp_subroot='server/debug_chain_claude_entropy_sparse_long_horizon')
    exp_dirs = {
        # 'BucketBrigade_Redundancy_1_ec0_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_aucbb_red1_ec0.0',
        # 'BucketBrigade_Redundancy_1_ec0-001_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_aucbb_red1_ec0.001',
        # 'BucketBrigade_Redundancy_2_ec0_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_aucbb_red2_ec0.0',
        # 'BucketBrigade_Redundancy_2_ec0-001_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_aucbb_red2_ec0.001',
        # 'Vickrey_Redundancy_1_ec0_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_aucv_red1_ec0.0',
        # 'Vickrey_Redundancy_1_ec0-001_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_aucv_red1_ec0.001',
        # 'Vickrey_Redundancy_2_ec0_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_aucv_red2_ec0.0',
        # 'Vickrey_Redundancy_2_ec0-001_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_aucv_red2_ec0.001',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='Chain_Sparse_Redundancy_Long_Horizon_gamma0-99', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    for fname, exp_dir in exp_dirs.items():
        p.load_plot_all_state_metrics(
            fname=fname, 
            exp_dir=exp_dir, 
            metrics=['mean_payoff', 'mean_bid'])

    exp_dirs = {
        # 'BucketBrigade_Redundancy_1_ec0_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_aucbb_red1_ec0.0',
        # 'BucketBrigade_Redundancy_1_ec0-001_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_aucbb_red1_ec0.001',
        # 'BucketBrigade_Redundancy_2_ec0_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_aucbb_red2_ec0.0',
        # 'BucketBrigade_Redundancy_2_ec0-001_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_aucbb_red2_ec0.001',
        # 'Vickrey_Redundancy_1_ec0_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_aucv_red1_ec0.0',
        'Vickrey_Redundancy_1_ec0-001_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_aucv_red1_ec0.001',
        # 'Vickrey_Redundancy_2_ec0_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_aucv_red2_ec0.0',
        'Vickrey_Redundancy_2_ec0-001_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_aucv_red2_ec0.001',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='Chain_Sparse_Redundancy_Long_Horizon_gamma1', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    for fname, exp_dir in exp_dirs.items():
        p.load_plot_all_state_metrics(
            fname=fname, 
            exp_dir=exp_dir, 
            metrics=['mean_payoff', 'mean_bid'])



def plot_1_29_20_CW6_long_horizon_stepcost():
    """ 
        Here we hope to recover the original experiment that worked. Then we will try to break it.
    """
    p = MultiAgentCurvePlotter(exp_subroot='server/debug_chain_claude_entropy_stepcost-01_horizonx4')
    exp_dirs = {
        'BucketBrigade_Redundancy_1_ec0_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr-0.1_aucbb_red1_ec0.0',
        # 'BucketBrigade_Redundancy_1_ec0-1_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr-0.1_aucbb_red1_ec0.1',
        'BucketBrigade_Redundancy_2_ec0_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr-0.1_aucbb_red2_ec0.0',
        # 'BucketBrigade_Redundancy_2_ec0-1_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr-0.1_aucbb_red2_ec0.1',
        'Vickrey_Redundancy_1_ec0_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr-0.1_aucv_red1_ec0.0',
        'Vickrey_Redundancy_1_ec0-1_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr-0.1_aucv_red1_ec0.1',
        'Vickrey_Redundancy_2_ec0_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr-0.1_aucv_red2_ec0.0',
        'Vickrey_Redundancy_2_ec0-1_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr-0.1_aucv_red2_ec0.1',

        'CCVickrey_Redundancy_1_ec0_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr-0.1_aucccv_red1_ec0.0',
        # 'CCVickrey_Redundancy_1_ec0-1_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr-0.1_aucccv_red1_ec0.1',
        'CCVickrey_Redundancy_2_ec0_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr-0.1_aucccv_red2_ec0.0',
        # 'CCVickrey_Redundancy_2_ec0-1_gamma0-99': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr-0.1_aucccv_red2_ec0.1',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='Chain_StepCost-01_Redundancy_Horizonx4_gamma0-99', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    # for fname, exp_dir in exp_dirs.items():
    #     p.load_plot_all_state_metrics(
    #         fname=fname, 
    #         exp_dir=exp_dir, 
    #         metrics=['mean_payoff', 'mean_bid'])

    exp_dirs = {
        'BucketBrigade_Redundancy_1_ec0_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_elc4_sr-0.1_aucbb_red1_ec0.0',
        # 'BucketBrigade_Redundancy_1_ec0-1_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_elc4_sr-0.1_aucbb_red1_ec0.1',
        'BucketBrigade_Redundancy_2_ec0_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_elc4_sr-0.1_aucbb_red2_ec0.0',
        # 'BucketBrigade_Redundancy_2_ec0-1_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_elc4_sr-0.1_aucbb_red2_ec0.1',

        'Vickrey_Redundancy_1_ec0_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_elc4_sr-0.1_aucv_red1_ec0.0',
        'Vickrey_Redundancy_1_ec0-1_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_elc4_sr-0.1_aucv_red1_ec0.1',
        'Vickrey_Redundancy_2_ec0_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_elc4_sr-0.1_aucv_red2_ec0.0',
        'Vickrey_Redundancy_2_ec0-1_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_elc4_sr-0.1_aucv_red2_ec0.1',

        'CCVickrey_Redundancy_1_ec0_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_elc4_sr-0.1_aucccv_red1_ec0.0',
        # 'CCVickrey_Redundancy_1_ec0-1_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_elc4_sr-0.1_aucccv_red1_ec0.1',
        'CCVickrey_Redundancy_2_ec0_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_elc4_sr-0.1_aucccv_red2_ec0.0',
        # 'CCVickrey_Redundancy_2_ec0-1_gamma1': 'CW6_g1.0_plr4e-05_ppo_h16_elc4_sr-0.1_aucccv_red2_ec0.1',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='Chain_StepCost-01_Redundancy_Horizonx4_gamma1', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    # for fname, exp_dir in exp_dirs.items():
    #     p.load_plot_all_state_metrics(
    #         fname=fname, 
    #         exp_dir=exp_dir, 
    #         metrics=['mean_payoff', 'mean_bid'])



def plot_1_30_20_CW6_shorthorizon_compare_ccv():
    """ 
        Compare CCV with bucket brigade 
        eplen=4
        step_reward=0
        gamma=0.99
    """
    p = MultiAgentCurvePlotter(exp_subroot='debug')
    exp_dirs = {

         'CCV red 2 cln': 'CW6_g0.99_plr4e-05_ppo_h16_cln_elc4_sr0.0_aucccv_red2_ec0.1',
         'CCV red 2': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucccv_red2_ec0.1',
         'CCV red 1': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucccv_red1_ec0.1',

         'BB red 2 cln': 'CW6_g0.99_plr4e-05_ppo_h16_cln_elc4_sr0.0_aucbb_red2_ec0.1',
         'BB red 2': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucbb_red2_ec0.1',
         'BB red 1': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucbb_red1_ec0.1',



        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='Chain_StepCost0_Eplen4_Redundancy_gamma0-99', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    for fname, exp_dir in exp_dirs.items():
        p.load_plot_all_state_metrics(
            fname=fname, 
            exp_dir=exp_dir, 
            metrics=['mean_payoff', 'mean_bid'])


def plot_1_30_20_CW6_gcp_chain():
    """ 
        horizon length 4
        step_reward 0
        
        bb v ccv 
        bb-clone v-clone ccv-clone
    """
    p = MultiAgentCurvePlotter(exp_subroot='server/chain_gcp')
    exp_dirs = {
        'Bucket Brigade Clone ec 0': 'CW6_g0.99_plr4e-05_ppo_h16_cln_elc4_sr0.0_aucbb_red2_ec0.0',
        'Credit Conserving Vickrey Clone ec 0': 'CW6_g0.99_plr4e-05_ppo_h16_cln_elc4_sr0.0_aucccv_red2_ec0.0',
        'Vickrey Clone ec 0': 'CW6_g0.99_plr4e-05_ppo_h16_cln_elc4_sr0.0_aucv_red2_ec0.0',

        'Bucket Brigade Red 1 ec 0': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucbb_red1_ec0.0',  # can finish
        'Bucket Brigade Red 2 ec 0': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucbb_red2_ec0.0',
        'Credit Conserving Vickrey Red 1 ec 0': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucccv_red1_ec0.0',  # can finish
        'Credit Conserving Vickrey Red 2 ec 0': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucccv_red2_ec0.0',
        'Vickrey Red 1 ec 0': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucv_red1_ec0.0',  # can finish
        'Vickrey Red 2 ec 0': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucv_red2_ec0.0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='Chain_GCP ec 0', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    # for fname, exp_dir in exp_dirs.items():
    #     p.load_plot_all_state_metrics(
    #         fname=fname, 
    #         exp_dir=exp_dir, 
    #         metrics=['mean_payoff', 'mean_bid'])

    exp_dirs = {
        'Bucket Brigade Clone ec 0.1': 'CW6_g0.99_plr4e-05_ppo_h16_cln_elc4_sr0.0_aucbb_red2_ec0.1',
        'Credit Conserving Vickrey Clone ec 0.1': 'CW6_g0.99_plr4e-05_ppo_h16_cln_elc4_sr0.0_aucccv_red2_ec0.1',
        'Vickrey Clone ec 0.1': 'CW6_g0.99_plr4e-05_ppo_h16_cln_elc4_sr0.0_aucv_red2_ec0.1',

        'Bucket Brigade Red 1 ec 0.1': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucbb_red1_ec0.1',  # can finish
        'Bucket Brigade Red 2 ec 0.1': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucbb_red2_ec0.1',
        'Credit Conserving Vickrey Red 1 ec 0.1': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucccv_red1_ec0.1',  # can finish
        'Credit Conserving Vickrey Red 2 ec 0.1': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucccv_red2_ec0.1',
        'Vickrey Red 1 ec 0.1': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucv_red1_ec0.1',  # can finish
        'Vickrey Red 2 ec 0.1': 'CW6_g0.99_plr4e-05_ppo_h16_elc4_sr0.0_aucv_red2_ec0.1',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='Chain_GCP ec 0.1', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    # for fname, exp_dir in exp_dirs.items():
    #     p.load_plot_all_state_metrics(
    #         fname=fname, 
    #         exp_dir=exp_dir, 
    #         metrics=['mean_payoff', 'mean_bid'])





def plot_1_31_20_CW6_gcp_lunarlander():
    """ 
        lunar lander

        Observations: seems to be plateauing really hard.
        All seem to be performing the same
    """
    p = CurvePlotter(exp_subroot='server/lunarlander')
    exp_dirs = {
        'Bucket Brigade Redundancy 1': 'LL-2_g0.99_plr4e-05_ppo_h32_aucbb_red1_ec0',
        'Bucket Brigade Redundancy 2': 'LL-2_g0.99_plr4e-05_ppo_h32_aucbb_red2_ec0',
        'Bucket Brigade Clone': 'LL-2_g0.99_plr4e-05_ppo_h32_cln_aucbb_red2_ec0',

        'Credit Conserving Vickrey Redundancy 1': 'LL-2_g0.99_plr4e-05_ppo_h32_aucccv_red1_ec0',
        'Credit Conserving Vickrey Redundancy 2': 'LL-2_g0.99_plr4e-05_ppo_h32_aucccv_red2_ec0',
        'Credit Conserving Vickrey Clone': 'LL-2_g0.99_plr4e-05_ppo_h32_cln_aucccv_red2_ec0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='LunarLander_GCP', 
                stats_dict=stats_dict, mode=mode, metric=metric)


def plot_1_31_20_CW6_gcp_babyai():
    """ 
        babyai

    """
    p = CurvePlotter(exp_subroot='server/babyai')
    exp_dirs = {

        'Bucket Brigade Redundancy 1': 'BAI-PK-0_g0.99_plr4e-05_ppo_h16_aucbb_red1_ec0',
        'Bucket Brigade Redundancy 2': 'BAI-PK-0_g0.99_plr4e-05_ppo_h16_aucbb_red2_ec0',
        'Bucket Brigade Clone': 'BAI-PK-0_g0.99_plr4e-05_ppo_h16_cln_aucbb_red2_ec0',

        'Credit Conserving Vickrey Redundancy 1': 'BAI-PK-0_g0.99_plr4e-05_ppo_h16_aucccv_red1_ec0',
        'Credit Conserving Vickrey Redundancy 2': 'BAI-PK-0_g0.99_plr4e-05_ppo_h16_aucccv_red2_ec0',
        'Credit Conserving Vickrey Clone': 'BAI-PK-0_g0.99_plr4e-05_ppo_h16_cln_aucccv_red2_ec0',

        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='BabyAI_PickupKey_GCP', 
                stats_dict=stats_dict, mode=mode, metric=metric)




    p = CurvePlotter(exp_subroot='server/babyai')
    exp_dirs = {

        'Bucket Brigade Redundancy 1': 'BAI-ODD-0_g0.99_plr4e-05_ppo_h16_aucbb_red1_ec0',
        'Bucket Brigade Redundancy 2': 'BAI-ODD-0_g0.99_plr4e-05_ppo_h16_aucbb_red2_ec0',
        'Bucket Brigade Clone': 'BAI-ODD-0_g0.99_plr4e-05_ppo_h16_cln_aucbb_red2_ec0',

        'Credit Conserving Vickrey Redundancy 1': 'BAI-ODD-0_g0.99_plr4e-05_ppo_h16_aucccv_red1_ec0',
        'Credit Conserving Vickrey Redundancy 2': 'BAI-ODD-0_g0.99_plr4e-05_ppo_h16_aucccv_red2_ec0',
        'Credit Conserving Vickrey Clone': 'BAI-ODD-0_g0.99_plr4e-05_ppo_h16_cln_aucccv_red2_ec0',

        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='BabyAI_OpenDoorDebug_GCP', 
                stats_dict=stats_dict, mode=mode, metric=metric)





def plot_1_31_20_CE1_geb():
    """ 
    """
    p = MultiAgentCurvePlotter(exp_subroot='server/exception1')
    exp_dirs = {
        'Bucket Brigade Redundancy 1': 'CE1_g0.99_plr4e-05_ppo_h16_aucbb_red1_ec0.0',
        'Bucket Brigade Redundancy 2': 'CE1_g0.99_plr4e-05_ppo_h16_aucbb_red2_ec0.0',
        'Credit Conserving Vickrey Redundancy 1': 'CE1_g0.99_plr4e-05_ppo_h16_aucccv_red1_ec0.0',
        'Credit Conserving Vickrey Redundancy 2': 'CE1_g0.99_plr4e-05_ppo_h16_aucccv_red2_ec0.0',
        'Vickrey Redundancy 1': 'CE1_g0.99_plr4e-05_ppo_h16_aucv_red1_ec0.0',
        'Vickrey Redundancy 2': 'CE1_g0.99_plr4e-05_ppo_h16_aucv_red2_ec0.0',
        'Bucket Brigade Clone': 'CE1_g0.99_plr4e-05_ppo_h16_cln_aucbb_red2_ec0.0',
        'Credit Conserving Vickrey Clone': 'CE1_g0.99_plr4e-05_ppo_h16_cln_aucccv_red2_ec0.0',
        'Vickrey Clone': 'CE1_g0.99_plr4e-05_ppo_h16_cln_aucv_red2_ec0.0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='Exception1', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    for fname, exp_dir in exp_dirs.items():
        p.load_plot_all_state_metrics(
            fname=fname, 
            exp_dir=exp_dir, 
            metrics=['mean_payoff', 'mean_bid'])


def plot_2_1_20_memoryless():
    """ 
        CE1-1 is more clear cut than CE1

        What about CE1-2?

        It turns out that this isn't actually memoryless!


    """
    p = MultiAgentCurvePlotter(exp_subroot='debug_memoryless')
    exp_dirs = {
        'CE1 CCV Red 1 gamma 0.99': 'CE1_g0.99_plr4e-05_ppo_h16_aucccv_red1_ec0.0',
        'CE1 CCV Red 2 clone gamma 0.99': 'CE1_g0.99_plr4e-05_ppo_h16_cln_aucccv_red2_ec0.0',
        'CE1 CCV Red 1 gamma 1': 'CE1_g1.0_plr4e-05_ppo_h16_aucccv_red1_ec0.0',
        'CE1 CCV Red 2 clone gamma 1': 'CE1_g1.0_plr4e-05_ppo_h16_cln_aucccv_red2_ec0.0',
        }

    # stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    # for mode in ['train', 'test']:
    #     for metric in ['mean_return', 'min_return', 'max_return']:
    #         p.plot_episode_metrics(fname='memoryless_ce1', 
    #             stats_dict=stats_dict, mode=mode, metric=metric)

    # for fname, exp_dir in exp_dirs.items():
    #     p.load_plot_all_state_metrics(
    #         fname=fname, 
    #         exp_dir=exp_dir, 
    #         metrics=['mean_payoff', 'mean_bid'])

    p = MultiAgentCurvePlotter(exp_subroot='debug_memoryless')
    exp_dirs = {
        'CE1-1 CCV Red 1 gamma 0.99': 'CE1-1_g0.99_plr4e-05_ppo_h16_aucccv_red1_ec0.0',
        'CE1-1 CCV Red 2 clone gamma 0.99': 'CE1-1_g0.99_plr4e-05_ppo_h16_cln_aucccv_red2_ec0.0',
        'CE1-1 CCV Red 1 gamma 1': 'CE1-1_g1.0_plr4e-05_ppo_h16_aucccv_red1_ec0.0',
        'CE1-1 CCV Red 2 clone gamma 1': 'CE1-1_g1.0_plr4e-05_ppo_h16_cln_aucccv_red2_ec0.0',
        }

    # stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    # for mode in ['train', 'test']:
    #     for metric in ['mean_return', 'min_return', 'max_return']:
    #         p.plot_episode_metrics(fname='memoryless_ce1-1', 
    #             stats_dict=stats_dict, mode=mode, metric=metric)

    # for fname, exp_dir in exp_dirs.items():
    #     p.load_plot_all_state_metrics(
    #         fname=fname, 
    #         exp_dir=exp_dir, 
    #         metrics=['mean_payoff', 'mean_bid'])


    p = MultiAgentCurvePlotter(exp_subroot='debug_memoryless')
    exp_dirs = {
        'CE1-2 CCV Red 1 gamma 0.99': 'CE1-2_g0.99_plr4e-05_ppo_h16_aucccv_red1_ec0.0',
        'CE1-2 CCV Red 2 clone gamma 0.99': 'CE1-2_g0.99_plr4e-05_ppo_h16_cln_aucccv_red2_ec0.0',
        'CE1-2 CCV Red 1 gamma 1': 'CE1-2_g1.0_plr4e-05_ppo_h16_aucccv_red1_ec0.0',
        'CE1-2 CCV Red 2 clone gamma 1': 'CE1-2_g1.0_plr4e-05_ppo_h16_cln_aucccv_red2_ec0.0',
        }

    # stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    # for mode in ['train', 'test']:
    #     for metric in ['mean_return', 'min_return', 'max_return']:
    #         p.plot_episode_metrics(fname='memoryless_ce1-2', 
    #             stats_dict=stats_dict, mode=mode, metric=metric)

    # for fname, exp_dir in exp_dirs.items():
    #     p.load_plot_all_state_metrics(
    #         fname=fname, 
    #         exp_dir=exp_dir, 
    #         metrics=['mean_payoff', 'mean_bid'])


    p = MultiAgentCurvePlotter(exp_subroot='server/debug_memoryless_geb')
    exp_dirs = {
        'GEB CE1 CCV Red 1 gamma 0.99': 'CE1_g0.99_plr4e-05_ppo_h16_aucccv_red1_ec0.0',
        'GEB CE1 CCV Red 2 clone gamma 0.99': 'CE1_g0.99_plr4e-05_ppo_h16_cln_aucccv_red2_ec0.0',
        'GEB CE1 CCV Red 1 gamma 1': 'CE1_g1.0_plr4e-05_ppo_h16_aucccv_red1_ec0.0',
        'GEB CE1 CCV Red 2 clone gamma 1': 'CE1_g1.0_plr4e-05_ppo_h16_cln_aucccv_red2_ec0.0',
        }

    # stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    # for mode in ['train', 'test']:
    #     for metric in ['mean_return', 'min_return', 'max_return']:
    #         p.plot_episode_metrics(fname='memoryless_ce1_geb', 
    #             stats_dict=stats_dict, mode=mode, metric=metric)

    # for fname, exp_dir in exp_dirs.items():
    #     p.load_plot_all_state_metrics(
    #         fname=fname, 
    #         exp_dir=exp_dir, 
    #         metrics=['mean_payoff', 'mean_bid'])


    p = MultiAgentCurvePlotter(exp_subroot='server/debug_memoryless_asym_geb')
    exp_dirs = {
        'GEB CE1-2 CCV Red 1 gamma 0.99': 'CE1-2_g0.99_plr4e-05_ppo_h16_aucccv_red1_ec0.0',
        'GEB CE1-2 CCV Red 2 clone gamma 0.99': 'CE1-2_g0.99_plr4e-05_ppo_h16_cln_aucccv_red2_ec0.0',
        'GEB CE1-2 CCV Red 1 gamma 1': 'CE1-2_g1.0_plr4e-05_ppo_h16_aucccv_red1_ec0.0',
        'GEB CE1-2 CCV Red 2 clone gamma 1': 'CE1-2_g1.0_plr4e-05_ppo_h16_cln_aucccv_red2_ec0.0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='memoryless_ce1-2_geb', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    for fname, exp_dir in exp_dirs.items():
        p.load_plot_all_state_metrics(
            fname=fname, 
            exp_dir=exp_dir, 
            metrics=['mean_payoff', 'mean_bid'])


def plot_1_30_20_CW6_gcp_chain_long():
    """ 
        horizon length 20
        step_reward 0
        
        bb v ccv 
        bb-clone v-clone ccv-clone



    """
    p = MultiAgentCurvePlotter(exp_subroot='server/chain_gcp_long')
    exp_dirs = {

            'G099_Bucket_Brigade_Red1': 'CW6_g0.99_plr4e-05_ppo_h16_elc20_sr0.0_aucbb_red1_ec0.0',
            'G099_Bucket_Brigade_Red2': 'CW6_g0.99_plr4e-05_ppo_h16_elc20_sr0.0_aucbb_red2_ec0.0',
            'G099_Bucket_Brigade_Red2_clone': 'CW6_g0.99_plr4e-05_ppo_h16_cln_elc20_sr0.0_aucbb_red2_ec0.0',
            'G099_CreditConserving_Vickrey_Red1': 'CW6_g0.99_plr4e-05_ppo_h16_elc20_sr0.0_aucccv_red1_ec0.0',
            'G099_CreditConserving_Vickrey_Red2': 'CW6_g0.99_plr4e-05_ppo_h16_elc20_sr0.0_aucccv_red2_ec0.0',
            'G099_CreditConserving_Vickrey_Red2_clone': 'CW6_g0.99_plr4e-05_ppo_h16_cln_elc20_sr0.0_aucccv_red2_ec0.0',
            # 'G099_Vickrey_Red1': 'CW6_g0.99_plr4e-05_ppo_h16_elc20_sr0.0_aucv_red1_ec0.0',
            # 'G099_Vickrey_Red2': 'CW6_g0.99_plr4e-05_ppo_h16_elc20_sr0.0_aucv_red2_ec0.0',
            # 'G099_Vickrey_Red2_clone': 'CW6_g0.99_plr4e-05_ppo_h16_cln_elc20_sr0.0_aucv_red2_ec0.0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='Chain_GCP_long_gamma099', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    for fname, exp_dir in exp_dirs.items():
        p.load_plot_all_state_metrics(
            fname=fname, 
            exp_dir=exp_dir, 
            metrics=['mean_payoff', 'mean_bid'])

    exp_dirs = {

            'G1_Bucket_Brigade_Red1': 'CW6_g1.0_plr4e-05_ppo_h16_elc20_sr0.0_aucbb_red1_ec0.0',
            'G1_Bucket_Brigade_Red2': 'CW6_g1.0_plr4e-05_ppo_h16_elc20_sr0.0_aucbb_red2_ec0.0',
            'G1_Bucket_Brigade_Red2_clone': 'CW6_g1.0_plr4e-05_ppo_h16_cln_elc20_sr0.0_aucbb_red2_ec0.0',
            'G1_CreditConserving_Vickrey_Red1': 'CW6_g1.0_plr4e-05_ppo_h16_elc20_sr0.0_aucccv_red1_ec0.0',
            'G1_CreditConserving_Vickrey_Red2': 'CW6_g1.0_plr4e-05_ppo_h16_elc20_sr0.0_aucccv_red2_ec0.0',
            'G1_CreditConserving_Vickrey_Red2_clone': 'CW6_g1.0_plr4e-05_ppo_h16_cln_elc20_sr0.0_aucccv_red2_ec0.0',
            # 'G1_Vickrey_Red1': 'CW6_g1.0_plr4e-05_ppo_h16_elc20_sr0.0_aucv_red1_ec0.0',
            # 'G1_Vickrey_Red2': 'CW6_g1.0_plr4e-05_ppo_h16_elc20_sr0.0_aucv_red2_ec0.0',
            # 'G1_Vickrey_Red2_clone': 'CW6_g1.0_plr4e-05_ppo_h16_cln_elc20_sr0.0_aucv_red2_ec0.0',

        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='Chain_GCP_long_gamma1', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    for fname, exp_dir in exp_dirs.items():
        p.load_plot_all_state_metrics(
            fname=fname, 
            exp_dir=exp_dir, 
            metrics=['mean_payoff', 'mean_bid'])



# now plot minigrid-empty



def plot_2_1_20_gcp_mg_empty():
    """ 
        mg_empty

    """
    p = CurvePlotter(exp_subroot='server/mg_empty')
    exp_dirs = {

            'BucketBrigade_Red_1': 'MG-E-R-55-0_g0.99_plr4e-05_ppo_h16_aucbb_red1_ec0',
            'BucketBrigade_Clone': 'MG-E-R-55-0_g0.99_plr4e-05_ppo_h16_cln_aucbb_red2_ec0',

            'CCVickrey_Red_1': 'MG-E-R-55-0_g0.99_plr4e-05_ppo_h16_aucccv_red1_ec0',
            'CCVickrey_Clone': 'MG-E-R-55-0_g0.99_plr4e-05_ppo_h16_cln_aucccv_red2_ec0',

        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='MG_EmptyR5x5_GCP', 
                stats_dict=stats_dict, mode=mode, metric=metric)


def plot_2_1_20_actual_memoryless():
    """ 
        This ACTUALLY is memoryless


    """
    # p = MultiAgentCurvePlotter(exp_subroot='debug_actual_memoryless')
    # exp_dirs = {
    #     'CE1-2 CCV Red 2 gamma 0.99': 'CE1-2_g0.99_plr4e-05_ppo_h16_cln_mryls_aucccv_red2_ec0.0',
    #     'CE1-2 CCV Red 1 gamma 0.99': 'CE1-2_g0.99_plr4e-05_ppo_h16_mryls_aucccv_red1_ec0.0',
    #     'CE1-2 CCV Red 2 gamma 1': 'CE1-2_g1.0_plr4e-05_ppo_h16_cln_mryls_aucccv_red2_ec0.0',
    #     'CE1-2 CCV Red 1 gamma 1': 'CE1-2_g1.0_plr4e-05_ppo_h16_mryls_aucccv_red1_ec0.0',
    #     }

    # stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    # for mode in ['train', 'test']:
    #     for metric in ['mean_return', 'min_return', 'max_return']:
    #         p.plot_episode_metrics(fname='actual_memoryless_ce1-2', 
    #             stats_dict=stats_dict, mode=mode, metric=metric)

    # for fname, exp_dir in exp_dirs.items():
    #     p.load_plot_all_state_metrics(
    #         fname=fname, 
    #         exp_dir=exp_dir, 
    #         metrics=['mean_payoff', 'mean_bid'])


    # p = MultiAgentCurvePlotter(exp_subroot='debug_actual_memoryless_entropy')
    # exp_dirs = {
    #     'EC1 CE1-2 CCV Red 2 gamma 0.99': 'CE1-2_g0.99_plr4e-05_ppo_h16_cln_mryls_aucccv_red2_ec1.0',
    #     'EC1 CE1-2 CCV Red 1 gamma 0.99': 'CE1-2_g0.99_plr4e-05_ppo_h16_mryls_aucccv_red1_ec1.0',
    #     'EC1 CE1-2 CCV Red 2 gamma 1': 'CE1-2_g1.0_plr4e-05_ppo_h16_cln_mryls_aucccv_red2_ec1.0',
    #     'EC1 CE1-2 CCV Red 1 gamma 1': 'CE1-2_g1.0_plr4e-05_ppo_h16_mryls_aucccv_red1_ec1.0',
    #     }

    # stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    # for mode in ['train', 'test']:
    #     for metric in ['mean_return', 'min_return', 'max_return']:
    #         p.plot_episode_metrics(fname='actual_memoryless_ce1-2_ec1', 
    #             stats_dict=stats_dict, mode=mode, metric=metric)

    # for fname, exp_dir in exp_dirs.items():
    #     p.load_plot_all_state_metrics(
    #         fname=fname, 
    #         exp_dir=exp_dir, 
    #         metrics=['mean_payoff', 'mean_bid'])


    # exp_dirs = {
    #     'EC0-1 CE1-2 CCV Red 2 gamma 0.99': 'CE1-2_g0.99_plr4e-05_ppo_h16_cln_mryls_aucccv_red2_ec0.1',
    #     'EC0-1 CE1-2 CCV Red 1 gamma 0.99': 'CE1-2_g0.99_plr4e-05_ppo_h16_mryls_aucccv_red1_ec0.1',
    #     'EC0-1 CE1-2 CCV Red 2 gamma 1': 'CE1-2_g1.0_plr4e-05_ppo_h16_cln_mryls_aucccv_red2_ec0.1',
    #     'EC0-1 CE1-2 CCV Red 1 gamma 1': 'CE1-2_g1.0_plr4e-05_ppo_h16_mryls_aucccv_red1_ec0.1',
    #     }

    # stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    # for mode in ['train', 'test']:
    #     for metric in ['mean_return', 'min_return', 'max_return']:
    #         p.plot_episode_metrics(fname='actual_memoryless_ce1-2_ec0-1', 
    #             stats_dict=stats_dict, mode=mode, metric=metric)

    # for fname, exp_dir in exp_dirs.items():
    #     p.load_plot_all_state_metrics(
    #         fname=fname, 
    #         exp_dir=exp_dir, 
    #         metrics=['mean_payoff', 'mean_bid'])

    p = MultiAgentCurvePlotter(exp_subroot='debug_actual_memoryless_entropy_tighter')
    exp_dirs = {
        'CCV Red 2 clone gamma 099': 'CE1-3_g0.99_plr4e-05_ppo_h16_cln_mryls_aucccv_red2_ec0.0',
        'CCV Red 1 gamma 099': 'CE1-3_g0.99_plr4e-05_ppo_h16_mryls_aucccv_red1_ec0.0',
        'CCV Red 2 clone gamma 1': 'CE1-3_g1.0_plr4e-05_ppo_h16_cln_mryls_aucccv_red2_ec0.0',
        'CCV Red 1 gamma 1': 'CE1-3_g1.0_plr4e-05_ppo_h16_mryls_aucccv_red1_ec0.0',

        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='actual_memoryless_ce1-3', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    for fname, exp_dir in exp_dirs.items():
        p.load_plot_all_state_metrics(
            fname=fname, 
            exp_dir=exp_dir, 
            metrics=['mean_payoff', 'mean_bid'])






def plot_2_1_20_CW6_gcp_chain_memoryless():
    """ 
        horizon length 4
        step_reward 0
        
        bb v ccv 
        bb-clone v-clone ccv-clone

        memoryless


    """
    p = MultiAgentCurvePlotter(exp_subroot='server/chain_gcp_memoryless')
    exp_dirs = {
        'Bucket_Brigade_Red_1_g099': 'CW6_g0.99_plr4e-05_ppo_h16_mryls_elc4_sr0.0_aucbb_red1_ec0.0',
        'Bucket_Brigade_Red_2_g099': 'CW6_g0.99_plr4e-05_ppo_h16_mryls_elc4_sr0.0_aucbb_red2_ec0.0',
        'Bucket_Brigade_Red_2_clone_g099': 'CW6_g0.99_plr4e-05_ppo_h16_cln_mryls_elc4_sr0.0_aucbb_red2_ec0.0',

        'Credit_Conserving_Vickrey_Red_1_g099': 'CW6_g0.99_plr4e-05_ppo_h16_mryls_elc4_sr0.0_aucccv_red1_ec0.0',
        'Credit_Conserving_Vickrey_Red_2_g099': 'CW6_g0.99_plr4e-05_ppo_h16_mryls_elc4_sr0.0_aucccv_red2_ec0.0',
        'Credit_Conserving_Vickrey_Red_2_clone_g099': 'CW6_g0.99_plr4e-05_ppo_h16_cln_mryls_elc4_sr0.0_aucccv_red2_ec0.0',

        'Vickrey_Red_1_g099': 'CW6_g0.99_plr4e-05_ppo_h16_mryls_elc4_sr0.0_aucv_red1_ec0.0',
        'Vickrey_Red_2_g099': 'CW6_g0.99_plr4e-05_ppo_h16_mryls_elc4_sr0.0_aucv_red2_ec0.0',
        'Vickrey_Red_2_clone_g099': 'CW6_g0.99_plr4e-05_ppo_h16_cln_mryls_elc4_sr0.0_aucv_red2_ec0.0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='Chain_GCP_memoryless_g099', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    for fname, exp_dir in exp_dirs.items():
        p.load_plot_all_state_metrics(
            fname=fname, 
            exp_dir=exp_dir, 
            metrics=['mean_payoff', 'mean_bid'])

    exp_dirs = {
        'Bucket_Brigade_Red_1_g1': 'CW6_g1.0_plr4e-05_ppo_h16_mryls_elc4_sr0.0_aucbb_red1_ec0.0',
        'Bucket_Brigade_Red_2_g1': 'CW6_g1.0_plr4e-05_ppo_h16_mryls_elc4_sr0.0_aucbb_red2_ec0.0',
        'Bucket_Brigade_Red_2_clone_g1': 'CW6_g1.0_plr4e-05_ppo_h16_cln_mryls_elc4_sr0.0_aucbb_red2_ec0.0',

        'Credit_Conserving_Vickrey_Red_1_g1': 'CW6_g1.0_plr4e-05_ppo_h16_mryls_elc4_sr0.0_aucccv_red1_ec0.0',
        'Credit_Conserving_Vickrey_Red_2_g1': 'CW6_g1.0_plr4e-05_ppo_h16_mryls_elc4_sr0.0_aucccv_red2_ec0.0',
        'Credit_Conserving_Vickrey_Red_2_clone_g1': 'CW6_g1.0_plr4e-05_ppo_h16_cln_mryls_elc4_sr0.0_aucccv_red2_ec0.0',

        'Vickrey_Red_1_g1': 'CW6_g1.0_plr4e-05_ppo_h16_mryls_elc4_sr0.0_aucv_red1_ec0.0',
        'Vickrey_Red_2_g1': 'CW6_g1.0_plr4e-05_ppo_h16_mryls_elc4_sr0.0_aucv_red2_ec0.0',
        'Vickrey_Red_2_clone_g1': 'CW6_g1.0_plr4e-05_ppo_h16_cln_mryls_elc4_sr0.0_aucv_red2_ec0.0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='Chain_GCP_memoryless_g1', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    for fname, exp_dir in exp_dirs.items():
        p.load_plot_all_state_metrics(
            fname=fname, 
            exp_dir=exp_dir, 
            metrics=['mean_payoff', 'mean_bid'])


# plot lunar lander memoryless
def plot_1_31_20_CW6_gcp_lunarlander_memoryless():
    """ 
        lunar lander: memoryless


    """
    p = CurvePlotter(exp_subroot='server/lunarlander')
    exp_dirs = {
            'BucketBrigade_Memoryless_Clone': 'LL-2_g0.99_plr4e-05_ppo_h32_cln_mryls_aucbb_red2_ec0',
            'CreditConservingVickrey_Memoryless_Clone': 'LL-2_g0.99_plr4e-05_ppo_h32_cln_mryls_aucccv_red2_ec0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='LunarLander_Memoryless', 
                stats_dict=stats_dict, mode=mode, metric=metric)

# plot babyai memoryless
def plot_2_2_20_gcp_babyai():
    """ 
        babyai

    """
    p = CurvePlotter(exp_subroot='server/babyai_memoryless')
    exp_dirs = {
        'CCV_Clone': 'BAI-PK-0_g0.99_plr4e-05_ppo_h16_cln_mryls_aucccv_red2_ec0',
        'CCV_Red1': 'BAI-PK-0_g0.99_plr4e-05_ppo_h16_mryls_aucccv_red1_ec0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='BabyAI_PickupKey_GCP_memoryless', 
                stats_dict=stats_dict, mode=mode, metric=metric)

    exp_dirs = {
        'CCV_Clone': 'BAI-ODD-0_g0.99_plr4e-05_ppo_h16_cln_mryls_aucccv_red2_ec0',
        'CCV_Red1': 'BAI-ODD-0_g0.99_plr4e-05_ppo_h16_mryls_aucccv_red1_ec0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='BabyAI_OpenDoorDebug_GCP_memoryless', 
                stats_dict=stats_dict, mode=mode, metric=metric)


# plot minigrid memoryless
def plot_2_2_20_gcp_mg88_memoryless():
    """ 
        mg88memoryless

    """
    p = CurvePlotter(exp_subroot='server/mg_empty8x8_memoryless')
    exp_dirs = {
        'CCV_Clone': 'MG-E-R-88-0_g0.99_plr4e-05_ppo_h16_cln_mryls_aucccv_red2_ec0',
        'CCV_Red1':  'MG-E-R-88-0_g0.99_plr4e-05_ppo_h16_mryls_aucccv_red1_ec0',
        }

    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='MinigridRandomEmpty8x8_GCP_memoryless', 
                stats_dict=stats_dict, mode=mode, metric=metric)



def plot_2_3_20_gcp_mg88_memoryless_sweep():
    """ 
        mg88memoryless

        Note that for each run, there are different vlrs that I have not shown.

    """
    # unorganized_exp_subroot = 'server/mg_empty8x8_memoryless_sweep'
    # unorganized_exp_dirs = {
    #     'plr1e-05': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_aucccv_red2_ec0',
    #     'plr1e-05_Memoryless': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_mryls_aucccv_red2_ec0',
    #     'plr1e-06': 'MG-E-R-88-0_g0.99_plr1e-06_ppo_h16_cln_aucccv_red2_ec0',
    #     'plr1e-06_Memoryless': 'MG-E-R-88-0_g0.99_plr1e-06_ppo_h16_cln_mryls_aucccv_red2_ec0',
    #     'plr2e-05': 'MG-E-R-88-0_g0.99_plr2e-05_ppo_h16_cln_aucccv_red2_ec0',
    #     'plr2e-05_Memoryless': 'MG-E-R-88-0_g0.99_plr2e-05_ppo_h16_cln_mryls_aucccv_red2_ec0',
    #     'plr3e-06': 'MG-E-R-88-0_g0.99_plr3e-06_ppo_h16_cln_aucccv_red2_ec0',
    #     'plr3e-06_Memoryless': 'MG-E-R-88-0_g0.99_plr3e-06_ppo_h16_cln_mryls_aucccv_red2_ec0',
    #     'plr5e-06': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_aucccv_red2_ec0',
    #     'plr5e-06_Memoryless': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_mryls_aucccv_red2_ec0',
    #     'pl8e-06': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_aucccv_red2_ec0',
    #     'pl8e-06_Memoryless': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_mryls_aucccv_red2_ec0',
    #     }

    # reorganize_folders(exp_subroot=unorganized_exp_subroot, exp_dirs=unorganized_exp_dirs.values())
    # assert False

    p = CurvePlotter(exp_subroot='server/mg_empty8x8_memoryless_sweep_reorg')

    # filter_pass 1: kill the bad max_return
    # exp_dirs = {
    #     # 'plr1e-05_vlr_0.001': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_aucccv_red2_ec0_vlr0.001', # maxreturn bad
    #     'plr1e-05_vlr_0.005': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',
    #     # 'plr1e-05_vlr_4e-05': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_aucccv_red2_ec0_vlr4e-05',  # maxreturn bad
    #     'plr1e-05_vlr_0.001_memoryless': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.001',
    #     'plr1e-05_vlr_0.005_memoryless': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.005',
    #     'plr1e-05_vlr_4e-05_memoryless': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr4e-05',

    #     # 'plr1e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr1e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',  # all bad
    #     # 'plr1e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr1e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',  # all bad
    #     # 'plr1e-06_vlr_4e-05': 'MG-E-R-88-0_g0.99_plr1e-06_ppo_h16_cln_aucccv_red2_ec0_vlr4e-05',  # all bad
    #     # 'plr1e-06_vlr_0.001_memoryless': 'MG-E-R-88-0_g0.99_plr1e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.001',  # all bad
    #     # 'plr1e-06_vlr_0.005_memoryless': 'MG-E-R-88-0_g0.99_plr1e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.005',  # all bad
    #     # 'plr1e-06_vlr_4e-05_memoryless': 'MG-E-R-88-0_g0.99_plr1e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr4e-05',  # all bad

    #     # 'plr2e-05_vlr_0.001': 'MG-E-R-88-0_g0.99_plr2e-05_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',  # too wild
    #     # 'plr2e-05_vlr_0.005': 'MG-E-R-88-0_g0.99_plr2e-05_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',  # too wild
    #     # 'plr2e-05_vlr_4e-05': 'MG-E-R-88-0_g0.99_plr2e-05_ppo_h16_cln_aucccv_red2_ec0_vlr4e-05',  # too wild
    #     # 'plr2e-05_vlr_0.001_memoryless': 'MG-E-R-88-0_g0.99_plr2e-05_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.001',  # too wild
    #     # 'plr2e-05_vlr_0.005_memoryless': 'MG-E-R-88-0_g0.99_plr2e-05_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.005',  # too wild
    #     # 'plr2e-05_vlr_4e-05_memoryless': 'MG-E-R-88-0_g0.99_plr2e-05_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr4e-05',  # too wild

    #     'plr3e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr3e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',
    #     'plr3e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr3e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',
    #     'plr3e-06_vlr_4e-05': 'MG-E-R-88-0_g0.99_plr3e-06_ppo_h16_cln_aucccv_red2_ec0_vlr4e-05',
    #     # 'plr3e-06_vlr_0.001_memoryless': 'MG-E-R-88-0_g0.99_plr3e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.001',  # max_return crashed
    #     # 'plr3e-06_vlr_0.005_memoryless': 'MG-E-R-88-0_g0.99_plr3e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.005',  # max_return crashed
    #     # 'plr3e-06_vlr_4e-05_memoryless': 'MG-E-R-88-0_g0.99_plr3e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr4e-05',  # max_return crashed

    #     'plr5e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',
    #     'plr5e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',
    #     'plr5e-06_vlr_4e-05': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_aucccv_red2_ec0_vlr4e-05',
    #     'plr5e-06_vlr_0.001_memoryless': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.001',
    #     'plr5e-06_vlr_0.005_memoryless': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.005',
    #     'plr5e-06_vlr_4e-05_memoryless': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr4e-05',

    #     'plr8e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',
    #     'plr8e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',
    #     # 'plr8e-06_vlr_4e-05': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_aucccv_red2_ec0_vlr4e-05',  # died
    #     'plr8e-06_vlr_0.001_memoryless': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.001',
    #     'plr8e-06_vlr_0.005_memoryless': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.005',
    #     'plr8e-06_vlr_4e-05_memoryless': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr4e-05',
    #     }

    # filter_pass 2: kill the bad mean_return
    # exp_dirs = {
    #     # # plr1e-05
    #     'plr1e-05_vlr_0.005': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',
    #     # 'plr1e-05_vlr_0.001_memoryless': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.001',  # meanreturn_unstable
    #     'plr1e-05_vlr_0.005_memoryless': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.005',
    #     'plr1e-05_vlr_4e-05_memoryless': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr4e-05',

    #      # plr1e-06

    #      # plr2e-05

    #     #  # plr3e-06
    #     'plr3e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr3e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',
    #     'plr3e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr3e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',
    #     'plr3e-06_vlr_4e-05': 'MG-E-R-88-0_g0.99_plr3e-06_ppo_h16_cln_aucccv_red2_ec0_vlr4e-05',

    #     # # plr5e-06
    #     'plr5e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',
    #     # 'plr5e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',  # meanreturn_unstable
    #     'plr5e-06_vlr_4e-05': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_aucccv_red2_ec0_vlr4e-05',
    #     'plr5e-06_vlr_0.001_memoryless': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.001',
    #     # 'plr5e-06_vlr_0.005_memoryless': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.005',  # meanreturn_unstable
    #     'plr5e-06_vlr_4e-05_memoryless': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr4e-05',

    #     # # plr8e-06
    #     'plr8e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',
    #     'plr8e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',
    #     # 'plr8e-06_vlr_0.001_memoryless': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.001',  # meanreturn_unstable
    #     # 'plr8e-06_vlr_0.005_memoryless': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.005',  # meanreturn_unstable
    #     'plr8e-06_vlr_4e-05_memoryless': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr4e-05',
    #     }

    # filter_pass 3: select the best min_return
    exp_dirs = {
        # # plr1e-05
        # 'plr1e-05_vlr_0.005': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',  # min_return unstable
        # 'plr1e-05_vlr_0.005_memoryless': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.005',  # min_return unstable
        # 'plr1e-05_vlr_4e-05_memoryless': 'MG-E-R-88-0_g0.99_plr1e-05_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr4e-05',  # min_return unstable

         # plr1e-06

         # plr2e-05

        #  # plr3e-06
        # 'plr3e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr3e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',  # min_return slower to converge
        # 'plr3e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr3e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',  # min_return slower to converge
        # 'plr3e-06_vlr_4e-05': 'MG-E-R-88-0_g0.99_plr3e-06_ppo_h16_cln_aucccv_red2_ec0_vlr4e-05',  # min_return slower to converge

        # # # plr5e-06
        # 'plr5e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',  # min_return has potential
        # 'plr5e-06_vlr_4e-05': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_aucccv_red2_ec0_vlr4e-05',  # min_return has potential
        # # 'plr5e-06_vlr_0.001_memoryless': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr0.001',
        # # 'plr5e-06_vlr_4e-05_memoryless': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr4e-05',

        # # # plr8e-06
        # 'plr8e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',  # best_min_return I've seen
        # 'plr8e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',  # best_min_return I've seen  # more stable  # best
        # # 'plr8e-06_vlr_4e-05_memoryless': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_mryls_aucccv_red2_ec0_vlr4e-05',
        }

    # filter_pass 4: these are the best
    # exp_dirs = {
    #     # plr1e-05

    #     # plr1e-06

    #     # plr2e-05

    #     # plr3e-06

    #     # plr5e-06
    #     'plr5e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',  # min_return has potential
    #     'plr5e-06_vlr_4e-05': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_aucccv_red2_ec0_vlr4e-05',  # min_return has potential

    #     # plr8e-06
    #     'plr8e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',  # best_min_return I've seen
    #     'plr8e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',  # best_min_return I've seen  # more stable  # best
    #     }


    # new runs
    exp_dirs = {
        # plr5e-06
        'plr5e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',  # min_return has potential
        'plr5e-06_vlr_4e-05': 'MG-E-R-88-0_g0.99_plr5e-06_ppo_h16_cln_aucccv_red2_ec0_vlr4e-05',  # min_return has potential

        # plr6e-06
        'plr6e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr6e-06_vlr0.001_ppo_h16_cln_aucccv_red2_ec0',
        'plr6e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr6e-06_vlr0.001_ppo_h16_cln_mryls_aucccv_red2_ec0',
        'plr6e-06_vlr_0.003': 'MG-E-R-88-0_g0.99_plr6e-06_vlr0.003_ppo_h16_cln_aucccv_red2_ec0',
        'plr6e-06_vlr_0.003': 'MG-E-R-88-0_g0.99_plr6e-06_vlr0.003_ppo_h16_cln_mryls_aucccv_red2_ec0',
        'plr6e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr6e-06_vlr0.005_ppo_h16_cln_aucccv_red2_ec0',
        'plr6e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr6e-06_vlr0.005_ppo_h16_cln_mryls_aucccv_red2_ec0',



        # plr7e-06

        'plr7e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr7e-06_vlr0.001_ppo_h16_cln_aucccv_red2_ec0',
        'plr7e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr7e-06_vlr0.001_ppo_h16_cln_mryls_aucccv_red2_ec0',
        'plr7e-06_vlr_0.003': 'MG-E-R-88-0_g0.99_plr7e-06_vlr0.003_ppo_h16_cln_aucccv_red2_ec0',
        'plr7e-06_vlr_0.003': 'MG-E-R-88-0_g0.99_plr7e-06_vlr0.003_ppo_h16_cln_mryls_aucccv_red2_ec0',
        'plr7e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr7e-06_vlr0.005_ppo_h16_cln_aucccv_red2_ec0',
        'plr7e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr7e-06_vlr0.005_ppo_h16_cln_mryls_aucccv_red2_ec0',


        # plr8e-06
        'plr8e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.001',  # best_min_return I've seen
        'plr8e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr8e-06_ppo_h16_cln_aucccv_red2_ec0_vlr0.005',  # best_min_return I've seen  # more stable  # best

        # plr9e-06

        'plr9e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr9e-06_vlr0.001_ppo_h16_cln_aucccv_red2_ec0',
        'plr9e-06_vlr_0.001': 'MG-E-R-88-0_g0.99_plr9e-06_vlr0.001_ppo_h16_cln_mryls_aucccv_red2_ec0',
        'plr9e-06_vlr_0.003': 'MG-E-R-88-0_g0.99_plr9e-06_vlr0.003_ppo_h16_cln_aucccv_red2_ec0',
        'plr9e-06_vlr_0.003': 'MG-E-R-88-0_g0.99_plr9e-06_vlr0.003_ppo_h16_cln_mryls_aucccv_red2_ec0',
        'plr9e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr9e-06_vlr0.005_ppo_h16_cln_aucccv_red2_ec0',
        'plr9e-06_vlr_0.005': 'MG-E-R-88-0_g0.99_plr9e-06_vlr0.005_ppo_h16_cln_mryls_aucccv_red2_ec0',


        }


    stats_dict = p.load_all_stats(exp_dirs=exp_dirs)
    for mode in ['train', 'test']:
        for metric in ['mean_return', 'min_return', 'max_return']:
            p.plot_episode_metrics(fname='MinigridRandomEmpty8x8_GCP_memoryless_sweep', 
                stats_dict=stats_dict, mode=mode, metric=metric)

# I should write some code to re-organize the folders
def reorganize_folders(exp_subroot, exp_dirs):
    """
    """
    exp_subroot = os.path.join(EXP_ROOT, exp_subroot)
    new_subroot = '{}_reorg'.format(exp_subroot)
    mkdirp(new_subroot)
    for exp_dir in exp_dirs:
        fp_exp_dir = os.path.join(exp_subroot, exp_dir)
        for seed_dir in [x for x in os.listdir(fp_exp_dir) if 'seed' in x]:
            fp_seed_dir = os.path.join(fp_exp_dir, seed_dir)
            # read json
            exp_dir_params = ujson.load(open(os.path.join(fp_seed_dir, 'code', 'params.json'), 'r'))
            # get the vlr value
            vlr = exp_dir_params['vlr']
            # get new exp_name
            new_exp_dir = exp_dir + '_vlr{}'.format(vlr)
            # make the new directory
            fp_new_exp_dir = os.path.join(new_subroot, new_exp_dir)
            mkdirp(fp_new_exp_dir)
            # copy subtree into new directory
            fp_new_seed_dir = os.path.join(fp_new_exp_dir, seed_dir)
            print('Copying {} to {}'.format(fp_seed_dir, fp_new_seed_dir))
            shutil.copytree(fp_seed_dir, fp_new_seed_dir)









if __name__ == '__main__':
    # plot_1_1_20_debug_return()
    # plot_1_1_20_debug_return_hdim32()


    # plot_vickrey_chain_debug_geb()
    # plot_1_2_20_debug_return_hdim16()


    # plot_english_chain_debug_geb()
    # plot_entropy_chain_debug_claude()


    # current
    # plot_bandit_claude()
    # plot_1_6_20_debug_lunarlander()
    # plot_1_7_20_CP_redundancy_1()

    # then actually we will just keep on iterating on these
    # now put these figures into the paper actually - programmatically.

    # 1/13/20
    # plot_1_13_20_CW6_sparse()
    # plot_1_13_20_debug_lunarlander_h()

    # plot_1_20_20_debug_reward_scaling()

    # plot_1_19_20_debug_minigrid_geb_h()

    # 1/27/20
    # plot_1_27_20_bandit_ado()
    # plot_1_27_20_debug_atari_breakout()

    # 1/28/20
    # plot_1_28_20_debug_babyai_sac()
    # plot_1_28_20_CW6_long_horizon()
    # plot_1_29_20_CW6_long_horizon_stepcost()

    # 1/30/20
    # plot_1_30_20_CW6_shorthorizon_compare_ccv()
    # plot_1_30_20_CW6_gcp_chain()

    # 1/31/20
    # plot_1_31_20_CW6_gcp_lunarlander()
    # plot_1_31_20_CW6_gcp_babyai()

    # plot_1_31_20_CE1_geb()
    plot_2_1_20_memoryless()  # <--
    # plot_1_30_20_CW6_gcp_chain_long()
    # plot_2_1_20_gcp_mg_empty()

    # thsi is for what is ACTUALLY memoryless
    # plot_2_1_20_actual_memoryless()
    plot_2_1_20_CW6_gcp_chain_memoryless()  # <--

    # plot lunar lander memoryless
    # plot_1_31_20_CW6_gcp_lunarlander_memoryless()

    # plot babyai memoryless
    # plot_2_2_20_gcp_babyai()

    # plot minigrid memoryless
    # plot_2_2_20_gcp_mg88_memoryless()


    plot_2_3_20_gcp_mg88_memoryless_sweep()

