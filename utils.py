import math

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

def visualize_parameters(model):
    for n, p in model.named_parameters():
        if p.grad is None:
            print(n, p.data.norm(), None)
        else:
            print(n, p.data.norm(), p.grad.data.norm())
