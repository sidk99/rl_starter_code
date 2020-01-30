from types import SimpleNamespace
from starter_code.env_config import simplify_name
from starter_code.log import MultiBaseLogger, MinigridEnvManager, GymEnvManager, TabularEnvManager
from starter_code.env_config import EnvRegistry
from starter_code.utils import AttrDict




"""
    global args
    local args
"""

def rlalg_config(args):
    rlalg_configs = dict(
        ppo=ppo_config,
        a2c=a2c_config,
        vpg=vpg_config,
        sac=sac_config,
    )
    args = rlalg_configs[args.alg_name](args)
    if not hasattr(args, 'parallel_update'):
        args.parallel_update = False
    return args

def sac_config(args):
    if not hasattr(args, 'gamma'):
        args.plr = 0.99
    if not hasattr(args, 'plr'):
        args.plr = 3e-4
    if not hasattr(args, 'vlr'):
        args.vlr = 3e-4
    if not hasattr(args, 'opt'):
        args.opt = 'adam'

    args.use_automatic_entropy_tuning = True
    if not hasattr(args, 'max_buffer_size'):
        args.max_buffer_size = int(1e6)  # override
    # args.num_trains_per_train_loop = 20#00  # number of epochs
    # args.num_samples_before_update = 1000
    args.num_trains_per_train_loop = 16#00  # number of epochs
    args.num_samples_before_update = 4096
    if args.debug:
        args.max_buffer_size = 300
        args.num_trains_per_train_loop = 5
        args.num_samples_before_update = 100
    return args

#   "algorithm": "SAC",
#   "version": "normal",
#   "layer_size": 256,
#   "replay_buffer_size": 1000000,
#   "algorithm_kwargs": {
#     "num_epochs": 3000,
#     "num_eval_steps_per_epoch": 5000,
#     "num_trains_per_train_loop": 1000,
#     "num_expl_steps_per_train_loop": 1000,
#     "min_num_steps_before_training": 1000,
#     "max_path_length": 1000,
#     "batch_size": 256
#   },
#   "trainer_kwargs": {
#     "soft_target_tau": 0.005,
#     "target_update_period": 1,
#     "reward_scale": 1,
#     "use_automatic_entropy_tuning": true
#   }

def ppo_config(args):
    if not hasattr(args, 'gamma'):
        args.plr = 0.99
    if not hasattr(args, 'plr'):
        args.plr = 4e-5
    if not hasattr(args, 'vlr'):
        args.vlr = 5e-3
    if not hasattr(args, 'opt'):
        args.opt = 'adam'
    if not hasattr(args, 'entropy_coeff'):
        args.entropy_coeff = 0
    return args

def a2c_config(args):
    if not hasattr(args, 'gamma'):
        args.plr = 0.99
    args.plr = 1e-4
    args.vlr = 5e-3
    args.opt = 'adam'
    return args

def vpg_config(args):
    if not hasattr(args, 'gamma'):
        args.plr = 0.99
    args.plr = 1e-4
    args.vlr = 5e-3
    args.opt = 'adam'
    return args

def experiment_config(args):
    args.gpu_index = 0
    args.log_every = 10
    args.save_every = 50
    if not hasattr(args, 'visualize_every'):
        args.visualize_every = 500
    if not hasattr(args, 'eval_every'):
        args.eval_every = 50
    args.max_epochs = int(1e7)
    args.num_test = 10
    if args.debug:
        args.max_epochs = 12
        args.eval_every = 3
        args.save_every = 3
        args.visualize_every = 3
        args.log_every = 3
        args.num_test = 4
    return args

def lifelong_config(args):
    args.parents = ['root']
    return args

def network_config(args):
    if not hasattr(args, 'hdim'):
        args.hdim = [16]
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
    args.expname += '_g{}'.format(args.gamma)
    args.expname += '_plr{}'.format(args.plr)
    args.expname += '_{}'.format(args.alg_name)
    args.expname += '_h{}'.format(args.hdim).replace('[','').replace(']','').replace(', ','-')
    print(args.env_name)
    if hasattr(args, 'clone'):
        args.expname += '_cln'
    if 'CW' in args.env_name[0]:
        assert len(args.env_name) == 1
        args.expname += '_elc{}_sr{}'.format(args.eplencoeff, args.step_reward)
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