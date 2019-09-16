from types import SimpleNamespace
from env_config import simplify_name

def rlalg_config_switch(alg_name):
    rlalg_configs = {
        'ppo': ppo_config,
        'a2c': a2c_config,
        'vpg': vpg_config
    }
    return rlalg_configs[alg_name]

def ppo_config(args):
    args.plr = 4e-5
    args.vlr = 5e-3
    args.opt = 'sgd'
    args.entropy_coeff = 0
    return args

def a2c_config(args):
    args.plr = 1e-4
    args.vlr = 5e-3
    args.opt = 'adam'
    return args

def vpg_config(args):
    args.plr = 1e-4
    args.vlr = 5e-3
    args.opt = 'adam'
    return args

def experiment_config(args):
    args.gpu_index = 0
    args.eval_every = 100
    args.log_every = 10
    if args.debug:
        pass
    return args

def training_config(args):
    args.anneal_policy_lr = True
    args.anneal_policy_lr_step = 100
    args.anneal_policy_lr_gamma = 0.99
    args.anneal_policy_lr_after = 500


    args.eval_every = 100
    args.log_every = 10
    return args

def build_expname(args):
    args.expname = simplify_name(args.env_name)
    args.expname += '_s{}'.format(args.seed)
    return args

def process_config(args):
    args = experiment_config(args)
    args = training_config(args)
    args = rlalg_config_switch(args.alg_name)(args)
    args = build_expname(args)
    return args