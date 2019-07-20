import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

class VickreyLogger(object):
    def __init__(self, env):
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
        plt.savefig('{}_bootstrap.png'.format(title.replace(' ', '_')))
        plt.close()  # plt.clf() shrinks the plot for some reason