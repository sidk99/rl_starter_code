import copy
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class RunningAverage(object):
    def __init__(self):
        super(RunningAverage, self).__init__()
        self.data = {}
        self.alpha = 0.01

    def update_variable(self, key, value):
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

class BaseLogger(object):
    def __init__(self, root, expname, setdate):
        self.root = root
        self.expname = expname
        self.logdir = self.create_logdir(root=self.root, expname=self.expname, setdate=setdate)

        self.data = {}
        self.metrics = {}

        self.run_avg = RunningAverage()

    def create_logdir(self, root, expname, setdate):
        self.logdir = os.path.join(root, expname)
        if setdate:
            if not expname == '': self.logdir += '-'
            self.logdir += '{date:%Y-%m-%d_%H-%M-%S}'.format(
            date=datetime.datetime.now())
        os.mkdir(self.logdir)
        return self.logdir

    def add_variables(self, names):
        for name in names:
            self.add_variable(name)

    def add_variable(self, name, include_running_avg=False):
        self.data[name] = []
        if include_running_avg:
            self.data['running_{}'.format(name)] = []

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

    def plot(self, var_pairs):
        for var1_name, var2_name in var_pairs:

            x_indices, x_values = zip(*self.data[var1_name])
            y_indices, y_values = zip(*self.data[var2_name])

            fname = '{}_{}'.format(self.expname, var2_name)
            plt.plot(x_values,y_values)
            plt.xlabel(var1_name)
            plt.ylabel(var2_name)
            plt.savefig(os.path.join(self.logdir,'{}.png'.format(fname)))
            plt.close()

class VickreyLogger(BaseLogger):
    def __init__(self, env, root, expname, setdate):
        super(VickreyLogger, self).__init__(root, expname, setdate)
        # so if you want to convert this to HDF5 you will need to pre-compute the size
        # so maybe a low priority for now
        self.i_episodes = []
        self.payoffs = {s: [] for s in env.states}
        self.bids = {s: [] for s in env.states}

    def record_episode(self, i_episode):
        self.i_episodes.append(i_episode)

    def record_payoffs(self, state, payoffs):
        self.payoffs[state].append(payoffs)

    def record_bids(self, state, bids):
        self.bids[state].append(bids)

    def visualize(self, state, title):
        payoffs = zip(*self.payoffs[state])
        bids = zip(*self.bids[state])

        for j, payoff in enumerate(payoffs):
            plt.plot(self.i_episodes, payoff, label='Payoff for agent {}'.format(j))
        for j, bid in enumerate(bids):
            plt.plot(self.i_episodes, bid, label='Bid for agent {}'.format(j))
        plt.legend()

        plt.title(title)
        plt.tight_layout()
        fname = '{}.png'.format(title.replace(' ', '_'))
        plt.savefig(os.path.join(self.logdir, fname))
        plt.close()  # plt.clf() shrinks the plot for some reason


