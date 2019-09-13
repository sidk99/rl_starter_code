def ppo_config(args):
    args.plr = 4e-5
    args.vlr = 5e-3
    args.anneal_policy_lr = True
    args.anneal_policy_lr_step = 100
    args.anneal_policy_lr_gamma = 0.99
    args.anneal_policy_lr_after = 500
    args.opt = 'sgd'
    args.eval_every = 10
    args.log_every = 1
    return args

def a2c_config(args):
    args.plr = 1e-4
    args.vlr = 5e-3
    args.opt = 'adam'
    args.eval_every = 100
    args.log_every = 10
    return args

def vpg_config(args):
    args.plr = 1e-4
    args.vlr = 5e-3
    args.opt = 'adam'
    args.eval_every = 100
    args.log_every = 10
    return args

# maybe you should spend some time doing the configs