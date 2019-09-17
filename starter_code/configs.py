from types import SimpleNamespace
from starter_code.env_config import simplify_name
from starter_code.log import MultiBaseLogger, MinigridEnvManager, GymEnvManager, TabularEnvManager

from env_config import EnvRegistry

def rlalg_config_switch(alg_name):
    rlalg_configs = {
        'ppo': ppo_config,
        'a2c': a2c_config,
        'vpg': vpg_config
    }
    return rlalg_configs[alg_name]

def ppo_config(args):
    args.gamma = 0.99
    args.plr = 4e-5
    args.vlr = 5e-3
    args.opt = 'sgd'
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
    args.eval_every = 100
    args.log_every = 10
    args.max_epochs = int(1e5)
    if args.debug:
        args.max_epochs = 30
        args.eval_every = 3
        args.log_every = 1
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
    if args.debug:
        args.expname += '_db'
    return args

def process_config(args):
    args = experiment_config(args)
    args = training_config(args)
    args = rlalg_config_switch(args.alg_name)(args)
    args = network_config(args)
    args = build_expname(args)
    return args

def env_manager_switch(env_name):
    er = EnvRegistry()
    envtype = er.get_env_type(env_name)
    env_manager = {
        'gym': GymEnvManager,
        'mg': MinigridEnvManager,
        'tab': TabularEnvManager
    }
    return env_manager[envtype]


        # args.max_episodes = 30
        # args.update_every = 5
        # args.eval_every = 10
        # args.log_every = 1
        # args.curr_every = 15