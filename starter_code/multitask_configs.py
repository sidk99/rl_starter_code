multitask_prog_specs = dict(
    mg_debug={
        0: dict(
            train=['MiniGrid-Empty-Random-5x5-v0'],
            test=['MiniGrid-Empty-Random-6x6-v0']),
        },
    gym_debug={
        0: dict(
            train=['CartPole-v0'],
            test=['CartPole-v0']),
        1: dict(
            train=['CartPole-v0'],
            test=['CartPole-v0'])
        },
    tab_debug={
        0: dict(
            train=['3s2t2af'],
            test=['3s2t2af']),
        1: dict(
            train=['3s2t2af'],
            test=['3s2t2af'])
        },
    mg_1_ahead={
        0: dict(
            train=['MiniGrid-Empty-Random-5x5-v0'],
            test=['MiniGrid-Empty-Random-6x6-v0']),
        1: dict(
            train=['MiniGrid-Empty-Random-6x6-v0'],
            test=['MiniGrid-Empty-Random-7x7-v0']),
        2: dict(
            train=['MiniGrid-Empty-Random-7x7-v0'],
            test=['MiniGrid-Empty-Random-8x8-v0']),
        3: dict(
            train=['MiniGrid-Empty-Random-8x8-v0'],
            test=['MiniGrid-Empty-Random-9x9-v0']),
        },
    mg_1_ahead_n_back={
        0: dict(
            train=['MiniGrid-Empty-Random-5x5-v0'],
            test=['MiniGrid-Empty-Random-6x6-v0']),
        1: dict(
            train=['MiniGrid-Empty-Random-6x6-v0'],
            test=['MiniGrid-Empty-Random-5x5-v0', 'MiniGrid-Empty-Random-7x7-v0']),
        2: dict(
            train=['MiniGrid-Empty-Random-7x7-v0'],
            test=['MiniGrid-Empty-Random-5x5-v0', 'MiniGrid-Empty-Random-6x6-v0', 'MiniGrid-Empty-Random-8x8-v0']),
        3: dict(
            train=['MiniGrid-Empty-Random-8x8-v0'],
            test=['MiniGrid-Empty-Random-5x5-v0', 'MiniGrid-Empty-Random-6x6-v0', 'MiniGrid-Empty-Random-7x7-v0', 'MiniGrid-Empty-Random-9x9-v0']),
        },
    )