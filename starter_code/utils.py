import copy
import math
import numpy as np
import torch
from operator import itemgetter

def to_device(device, *args):
    return [x.to(device) for x in args]

def display_stats(stats):
    metrics = list(stats.keys())
    max_metric_length = max(len(x) for x in metrics)
    aggregate_keys = list(stats[metrics[0]].keys())
    num_aggregates = len(aggregate_keys)
    agg_label_length = max(len(x) for x in stats[metrics[0]]) + 3
    ###############################################################
    value_length = 7
    pad = 2
    lefter_width = pad + max_metric_length + pad
    column_width = pad + agg_label_length + value_length + pad
    border_length = lefter_width + (column_width+1)*num_aggregates + 3
    ###############################################################
    doubledash = '=' * border_length
    dash = '-' * border_length
    display_str = '{}\n'.format(doubledash)
    header_str = '|{:^{width}s} '.format('', width=lefter_width)
    for a in sorted(aggregate_keys):
        header_str += '|{:^{width}}'.format(a, width=column_width)
    display_str += header_str +'|'
    display_str += '\n{}\n'.format(dash)
    ###############################################################
    for m in sorted(stats.keys()):
        metric_str = '|{:^{width}s} '.format(m, width=lefter_width)
        for a in sorted(stats[m].keys()):
            metric_str += '|{:^{width}.4f}'.format(stats[m][a], width=column_width)
        display_str += metric_str+'|\n'
    ###############################################################
    display_str += doubledash
    return display_str

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def visualize_parameters(model, pfunc):
    for n, p in model.named_parameters():
        if p.grad is None:
            pfunc('{}\t{}\t{}'.format(n, p.data.norm(), None))
        else:
            pfunc('{}\t{}\t{}'.format(n, p.data.norm(), p.grad.data.norm()))

def visualize_params(dict_of_models, pfunc):
    for k, v in dict_of_models.items():
        pfunc('#'*20 + ' {} '.format(k) + '#'*(80-len(k)-2-20))
        visualize_parameters(v, pfunc)
    pfunc('#'*80)

class TupleList(object):
    def __init__(self, list_of_tuples):
        """
            [(index, value), (index, value), ...]

            # Wait. This is just a dictionary.
        """
        self.list_of_tuples = list_of_tuples
        self.check_no_duplicates(self.list_of_tuples)

    def check_no_duplicates(self, list_of_tuples):
        items = set()
        for index, value in list_of_tuples:
            before_size = len(items)
            items.add(index)
            after_size = len(items)
            assert after_size == before_size + 1

    def __getitem__(self, key):
        for pos, element in enumerate(self.list_of_tuples):
            if element[0] == key:
                return copy.deepcopy(element[1])
        raise ValueError('index {} not in list'.format(key))

    def __str__(self):
        return str(copy.deepcopy(self.list_of_tuples))

    def index_of_max(self):
        argmax = max(self.list_of_tuples,key=itemgetter(1))[0] 
        return argmax

    def enumerate(self):
        for tup in self.list_of_tuples:
            yield (tup[0], tup[1])

    def indices(self):
        return [e[0] for e in self.list_of_tuples]

    def __len__(self):
        return len(self.list_of_tuples)


# # Stops iterating through the list as soon as it finds the value
def get_index_of_tuple(a_list, index_in_tuple, index_value):
    for pos,t in enumerate(a_list):
        if t[index_in_tuple] == index_value:
            return pos

    # Matches behavior of list.index
    raise ValueError("list.index(x): x not in list")

def to_onehot(state_id, state_dim):
    state = np.zeros(state_dim)
    state[state_id] = 1
    return state

def from_onehot(state, state_dim):
    state_id = np.argmax(state)
    if state_id == state_dim-1:
        state_id = -1
    return state_id

def all_same(items):
    return all(x == items[0] for x in items)

def flatten_dictionary_values_by_key(list_of_dicts):
    """
        assume all dicts list_of_dicts have the same keys
    """
    assert all_same(sorted(d.keys()) for d in list_of_dicts)
    
    pass



class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__    


def from_np(np_array, device):
    return torch.tensor([np_array]).float().to(device)

def to_np(tensor):
    return tensor.detach().cpu().numpy()

def is_float(n):
    return isinstance(n, float) or isinstance(n, np.float64) or isinstance(n, np.float32)


# create a switching data structure thatyou can use to index



def tranpose_list_of_dicts(list_of_dicts):
    """
    input:
        [
            {key1: val1_0, key2: val2_0, ...},
            {key1: val1_1, key2: val2_1, ...},
        ]

    output:
        {
            key1: [val1_0, val1_1, ...]
            key2: [val2_0, val2_1, ...]
        }
    """
    keys = list_of_dicts[0].keys()
    output = {key: [] for key in keys}
    for element_dict in list_of_dicts:
        for key in element_dict:
            output[key].append(element_dict[key])
    return output

def transpose_dict_of_lists(dict_of_lists):
    """
    input:
        {
            key1: [val1_0, val1_1, ...]
            key2: [val2_0, val2_1, ...]
        }

    output:
        [
            {key1: val1_0, key2: val2_0, ...},
            {key1: val1_1, key2: val2_1, ...},
        ]
    """




