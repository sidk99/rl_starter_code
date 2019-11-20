from types import SimpleNamespace
from starter_code.env_config import simplify_name
from starter_code.log import MultiBaseLogger, MinigridEnvManager, GymEnvManager, TabularEnvManager

from env_config import EnvRegistry
from starter_code.utils import AttrDict


"""
    global args
    local args
"""

def rlalg_config_switch(alg_name):
    rlalg_configs = dict(
        ppo = ppo_config,
        a2c = a2c_config,
        vpg = vpg_config,
    )
    return rlalg_configs[alg_name]

def rlalg_config(args):
    rlalg_configs = dict(
        ppo = ppo_config,
        a2c = a2c_config,
        vpg = vpg_config,
    )
    args = rlalg_configs[args.alg_name](args)
    if not hasattr(args, 'parallel_update'):
        args.parallel_update = False
    return args

def ppo_config(args):
    args.gamma = 0.99
    if not hasattr(args, 'plr'):
        args.plr = 4e-5
    if not hasattr(args, 'vlr'):
        args.vlr = 5e-3
    if not hasattr(args, 'opt'):
        args.opt = 'sgd'
    if not hasattr(args, 'entropy_coeff'):
        args.entropy_coeff = 0
    return args

def a2c_config(args):
    args.gamma = 0.99
    args.plr = 1e-4
    args.vlr = 5e-3
    args.opt = 'adam'
    return args

def vpg_config(args):
    args.gamma = 0.99
    args.plr = 1e-4
    args.vlr = 5e-3
    args.opt = 'adam'
    return args

def experiment_config(args):
    args.gpu_index = 0
    # args.eval_every = 5000
    # args.log_every = 100
    args.eval_every = 500
    args.log_every = 10
    args.visualize_every = 50
    args.max_epochs = int(1e7)
    args.num_test = 100
    if args.debug:
        args.max_epochs = 12
        args.eval_every = 3
        args.log_every = 3
        args.num_test = 10
    return args

def lifelong_config(args):
    args.parents = ['root']
    return args

def network_config(args):
    args.hdim = [128, 128]
    if args.debug:
        args.hdim = [20, 20]
    return args

def training_config(args):
    args.anneal_policy_lr = True
    args.anneal_policy_lr_step = 100
    args.anneal_policy_lr_gamma = 0.99
    args.anneal_policy_lr_after = 500
    if args.debug:
        args.anneal_policy_lr_step = 1
        args.anneal_policy_lr_after = 2
    return args

def build_expname(args):
    args.expname = simplify_name(args.env_name)
    args.expname += '_s{}'.format(args.seed)
    args.expname += '_plr{}'.format(args.plr)
    args.expname += '_opt{}'.format(args.opt)
    if args.debug:
        args.expname += '_db'
    if hasattr(args, 'auctiontype'):
        args.expname += '_auc{}'.format(args.auctiontype)
    if hasattr(args, 'ado'):
        if args.ado:
            args.expname += '_ado'
    if hasattr(args, 'redundancy'):
        args.expname += '_red{}'.format(args.redundancy)
    if hasattr(args, 'entropy_coeff'):
        args.expname += '_ec{}'.format(args.entropy_coeff)
    return args

def process_config(args):
    args = experiment_config(args)
    args = training_config(args)
    # args = rlalg_config_switch(args.alg_name)(args)
    args = rlalg_config(args)
    args = network_config(args)
    args = lifelong_config(args)
    args = build_expname(args)
    return args

def env_manager_switch(env_name, env_registry):
    envtype = env_registry.get_env_type(env_name)
    env_manager = dict(
        gym = GymEnvManager,
        mg = MinigridEnvManager,
        tab = TabularEnvManager,
    )
    return env_manager[envtype]