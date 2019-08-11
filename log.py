import copy
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shutil
import pprint

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

# class BaseLogger(object):
#     def __init__(self, args):
#         self.args = args
#         self.root = args.root
#         self.expname = args.expname
#         self.logdir = self.create_logdir(root=self.root, expname=self.expname, setdate=True)

#     def printf(self, string):
#         if self.args.printf:
#             f = open(os.path.join(self.logdir, self.expname+'.txt'), 'a')
#             print(string, file=f)
#         else:
#             print(string)

#     def pprintf(self, string):
#         if self.args.printf:
#             f = open(os.path.join(self.logdir, self.expname+'.txt'), 'a')
#             pprint.pprint(string, stream=f)
#         else:
#             pprint.pprint(string)

#     def create_logdir(self, root, expname, setdate):
#         self.logdir = os.path.join(root, expname)
#         if setdate:
#             if not expname == '': self.logdir += '-'
#             self.logdir += '{date:%Y-%m-%d_%H-%M-%S}'.format(
#             date=datetime.datetime.now())
#         os.mkdir(self.logdir)
#         return self.logdir

#     def remove_logdir(self):
#         should_remove = input('Remove {}? [y/n] '.format(self.logdir))
#         if should_remove == 'y':
#             shutil.rmtree(self.logdir)
#             print('Removed {}'.format(self.logdir))
#         else:
#             print('Did not remove {}'.format(self.logdir))

# class MultiBaseLogger(object):
#     def __init__(self, loggers, args):
#         self.loggers = loggers
#         self.args = args
#         self.root = args.root
#         self.expname = args.expname
#         self.logdir = self.create_logdir(root=self.root, expname=self.expname, setdate=True)

#     def printf(self, string):
#         if self.args.printf:
#             f = open(os.path.join(self.logdir, self.expname+'.txt'), 'a')
#             print(string, file=f)
#         else:
#             print(string)

#     def pprintf(self, string):
#         if self.args.printf:
#             f = open(os.path.join(self.logdir, self.expname+'.txt'), 'a')
#             pprint.pprint(string, stream=f)
#         else:
#             pprint.pprint(string)

#     def create_logdir(self, root, expname, setdate):
#         self.logdir = os.path.join(root, expname)
#         if setdate:
#             if not expname == '': self.logdir += '-'
#             self.logdir += '{date:%Y-%m-%d_%H-%M-%S}'.format(
#             date=datetime.datetime.now())
#         os.mkdir(self.logdir)
#         return self.logdir

#     def remove_logdir(self):
#         should_remove = input('Remove {}? [y/n] '.format(self.logdir))
#         if should_remove == 'y':
#             shutil.rmtree(self.logdir)
#             print('Removed {}'.format(self.logdir))
#         else:
#             print('Did not remove {}'.format(self.logdir))

#     def add_variable(self, env_name, name, incl_run_avg=False, metric=None):
#         self.loggers[env_name].add_variable(name, incl_run_avg, metric)

#     def update_variable(self, env_name, name, index, value, include_running_avg=False):
#         self.loggers[env_name].update_variable(name, index, value, include_running_avg)

#     def get_recent_variable_value(self, env_name, name):
#         self.loggers[env_name].get_recent_variable_value(name)

#     def add_metric(self, env_name, name, initial_val, comparator):
#         self.loggers[env_name].add_metric(name, initial_val, comparator)

#     def plot(self, env_name, var_pairs, logdir, expname):
#         self.loggers[env_name].plot(var_pairs, logdir, expname)


class MultiBaseLogger(object):
    def __init__(self, env_wrappers, args):
        self.env_wrappers = env_wrappers
        self.args = args
        self.root = args.root
        self.expname = args.expname
        self.logdir = self.create_logdir(root=self.root, expname=self.expname, setdate=True)

        for env_name, env_wrapper in self.env_wrappers.items():
            env_wrapper.set_logdir(self.create_logdir(root=self.logdir, expname=env_name, setdate=False))
        # now create the logdirs for all the envs

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

    def create_logdir(self, root, expname, setdate):
        logdir = os.path.join(root, expname)
        if setdate:
            if not expname == '': logdir += '-'
            logdir += '{date:%Y-%m-%d_%H-%M-%S}'.format(
            date=datetime.datetime.now())
        os.mkdir(logdir)
        return logdir

    def remove_logdir(self):
        should_remove = input('Remove {}? [y/n] '.format(self.logdir))
        if should_remove == 'y':
            shutil.rmtree(self.logdir)
            print('Removed {}'.format(self.logdir))
        else:
            print('Did not remove {}'.format(self.logdir))

    # def add_variable(self, env_name, name, incl_run_avg=False, metric=None):
    #     self.env_wrappers[env_name].env_logger.add_variable(name, incl_run_avg, metric)

    def get_recent_variable_value(self, env_name, name):
        self.env_wrappers[env_name].env_logger.get_recent_variable_value(name)

    # def add_metric(self, env_name, name, initial_val, comparator):
    #     self.env_wrappers[env_name].env_logger.add_metric(name, initial_val, comparator)


class EnvLogger(object):
    def __init__(self, args):
        super(EnvLogger, self).__init__()
        self.data = {}
        self.metrics = {}
        self.run_avg = RunningAverage()

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

    def plot(self, var_pairs, logdir, expname):
        for var1_name, var2_name in var_pairs:
            x_indices, x_values = zip(*self.data[var1_name])
            y_indices, y_values = zip(*self.data[var2_name])
            fname = '{}_{}'.format(expname, var2_name)
            plt.plot(x_values,y_values)
            plt.xlabel(var1_name)
            plt.ylabel(var2_name)
            plt.savefig(os.path.join(logdir, '{}.png'.format(fname)))
            plt.close()

class TabularEnvLogger(object):
    def __init__(self, args):
        super(TabularEnvLogger, self).__init__()