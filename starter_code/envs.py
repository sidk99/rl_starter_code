import itertools
import numpy as np
import pprint

class OneHotEnv(object):
    def __init__(self):
        super(OneHotEnv, self).__init__()

    @property
    def state_dim(self):
        return len(self.states)+1

    def to_onehot(self, state_id):
        state = np.zeros(self.state_dim)
        state[state_id] = 1
        return state

    def from_onehot(self, state):
        state_id = np.argmax(state)
        return state_id

def not_in_array(np_array, value):
    return not np.all(np.equal(np_array, value)) 


class GridWorldK(object):
    def __init__(self, height, width, goal, step_reward=-0.1):
        self.height = height
        self.width = width
        self.num_states = self.height * self.width
        self.goal_idx = self.get_goal_idx(goal)
        self.goal_state = (self.goal_idx // self.width, self.goal_idx % self.width)

        self.grid_states = self.create_grid_states()
        self.states = self.grid_states.flatten()
        self.actions = self.get_actions()#{0: 'L', 1: 'R', 2: 'D', 3: 'U'}
        self.transitions = self.create_transitions()

        self.step_reward = step_reward#-0.005#-0.1  # NOTE THAT THIS IS SPARSE REWARD!
        self.goal_reward = 0.8  # note that for GW2 I use 0.5 here
        self.rewards = self.create_rewards()

        print('grid_states', self.grid_states)
        print('states', self.states)
        print('action', self.actions)
        print('transitions', self.transitions)
        print('rewards', self.rewards)

        self.gamma = 1
        self.Qs = self.Q_iteration_deterministic()

    def get_actions(self):
        return {0: 'L', 1: 'R', 2: 'D', 3: 'U'}

    def get_goal_idx(self, goal):
        if goal is None:
            goal_idx = np.random.randint(self.num_states)
        elif goal == -1:
            goal_idx = self.num_states -1
        else:
            goal_idx = goal
        return goal_idx

    def create_grid_states(self):
        grid = np.arange(self.num_states, dtype=np.int64).reshape(self.height, self.width)
        grid[self.goal_state[0], self.goal_state[1]] = self.goal_idx  # goal
        return grid

    def create_transitions(self):
        transition_matrix = np.ones((len(self.actions), self.num_states), dtype=np.int64)*-2  # default value
        print(transition_matrix)
        for row_idx in range(self.height):
            for col_idx in range(self.width):
                state_id = self.grid_states[row_idx, col_idx]
                for action_id in self.actions:
                    if action_id == 0:  # L
                        if  col_idx == 0:
                            transition_matrix[action_id, state_id] = state_id  # don't move
                        else:
                            transition_matrix[action_id, state_id] = self.grid_states[row_idx, col_idx-1]  # left by one column
                    elif action_id == 1:  # R
                        if col_idx == self.width-1:
                            transition_matrix[action_id, state_id] = state_id  # don't move
                        else:
                            transition_matrix[action_id, state_id] = self.grid_states[row_idx, col_idx+1]  # right by one column
                    elif action_id == 2:  # D
                        if row_idx == self.height-1:
                            transition_matrix[action_id, state_id] = state_id  # don't move
                        else:
                            transition_matrix[action_id, state_id] = self.grid_states[row_idx+1, col_idx]  # down by one column
                    elif action_id == 3:   # U
                        if row_idx == 0:
                            transition_matrix[action_id, state_id] = state_id  # don't move
                        else:
                            transition_matrix[action_id, state_id] = self.grid_states[row_idx-1, col_idx]  # up by one column
                    else:
                        assert False

        # the goal state should transition to itself
        transition_matrix[:, self.goal_idx] = self.goal_idx
        assert not_in_array(transition_matrix, -2)
        return transition_matrix

    def create_rewards(self):
        reward_matrix = np.zeros((len(self.actions), self.num_states))
        for action_id in self.actions:
            for state_id in self.states:
                if self.transitions[action_id, state_id] == self.goal_idx:
                    reward_matrix[action_id, state_id] = self.goal_reward
                else:
                    reward_matrix[action_id, state_id] = self.step_reward
        reward_matrix[:, self.goal_idx] = 0
        return reward_matrix

    # this should be fine
    def Q_iteration_deterministic(self):
        Q_old = np.zeros((len(self.actions),self.num_states))
        Q_new = np.zeros((len(self.actions),self.num_states))

        i = 0
        while True:
            print('Q at iteration {}: \n{}'.format(i, Q_old))
            for s in self.states:
                for a in self.actions:
                    s_prime = self.transitions[a][s]
                    newQ = self.rewards[a][s] + self.gamma * np.max(Q_old[:, s_prime])
                    Q_new[a, s] = newQ
            if np.array_equal(Q_new, Q_old):
                print('Q converged to \n{} in {} iterations'.format(Q_new, i))
                break
            Q_old = np.copy(Q_new)
            i += 1
        return Q_old


def convert_grid_to_dict(grid):
    d = {}
    for action_idx in range(len(grid)):
        for state_idx in range(len(grid[action_idx])):
            reported_state_idx = state_idx
            d[reported_state_idx, action_idx] = grid[action_idx][state_idx]
    return d

class MultiStepEnv(OneHotEnv):
    def __init__(self):
        super(MultiStepEnv, self).__init__()
        self.states = []
        self.actions = []
        self.transitions = {}
        self.rewards = {}
        self.Qs = {}
        self.gamma = 1

    def getQ(self, state, action):
        state = self.from_onehot(state)
        return self.Qs[(state, action)]

    def reset(self):
        self.counter = 0
        start_state = np.random.choice(self.starting_states)
        self.state = start_state  # scalar
        return self.to_onehot(start_state)

    def step(self, action):
        next_state = self.transitions[self.state, action]
        self.counter += 1
        done = self.counter == self.eplen
        reward = self.rewards[self.state, action]
        self.state = next_state
        return self.to_onehot(next_state), reward, done, {}

    def render(self, mode):
        pass

    def close(self):
        pass

    def seed(self, seed):
        pass

class OneHotGridWorldK(GridWorldK, OneHotEnv):
    def __init__(self, height, width, rand_init=False, goal=-1, eplencoeff=4, step_reward=-0.1):
        super(OneHotGridWorldK, self).__init__(
            height=height, width=width, goal=goal, step_reward=step_reward)
        self.actions = list(self.actions.keys())
        self.transitions = convert_grid_to_dict(self.transitions)
        self.rewards = convert_grid_to_dict(self.rewards)
        self.Qs = convert_grid_to_dict(self.Qs)
        self.eplen = eplencoeff * (height-1 + width - 1)  # previously, eplencoeff=4, then I tried eplencoeff=20

        self.goal_states = [self.goal_idx]
        self.starting_states = self.get_initial_states(rand_init, self.goal_states)

    def get_initial_states(self, rand_init, goal_states):
        if rand_init:
            starting_states = [x for x in self.states if x not in goal_states]
        else:
            starting_states = [0]
        return starting_states
    
    def getQ(self, state, action):
        state = self.from_onehot(state)
        return self.Qs[(state, action)]

    def reset(self):
        self.counter = 0
        start_state = np.random.choice(self.starting_states)
        self.state = start_state  # scalar
        return self.to_onehot(start_state)

    def step(self, action):
        next_state = self.transitions[self.state, action]
        self.counter += 1
        done = next_state in self.goal_states or self.counter == self.eplen
        reward = self.rewards[self.state, action]
        self.state = next_state
        return self.to_onehot(next_state), reward, done, {}

    def render(self, mode):
        pass

    def close(self):
        pass

    def seed(self, seed):
        pass

class OneHotSquareGridWorldK(OneHotGridWorldK):
    def __init__(self, size, rand_init=False, goal=-1, eplencoeff=4, step_reward=-0.1):
        super(OneHotSquareGridWorldK, self).__init__(
            height=size, 
            width=size, 
            rand_init=rand_init, 
            goal=goal,
            eplencoeff=eplencoeff, 
            step_reward=step_reward
            )


class OneHotChainK(OneHotGridWorldK):
    def __init__(self, length, rand_init=False, goal=-1, eplencoeff=4, step_reward=-0.1):
        super(OneHotChainK, self).__init__(
            height=1, 
            width=length, 
            rand_init=rand_init, 
            goal=goal, 
            eplencoeff=eplencoeff, 
            step_reward=step_reward
            )

    def get_actions(self):
        return {0: 'L', 1: 'R'}


class OneStateOneStepKActionEnv(MultiStepEnv):
    def __init__(self, k):
        super(OneStateOneStepKActionEnv, self).__init__()
        self.states = [0]
        self.actions = list(range(k))
        self.transitions = {(0, kk): -1 for kk in range(k)}
        self.rewards = {(0, kk): (kk+1)*1.0/(k+1) for kk in range(k)}
        self.Qs = {**{(0, kk): (kk+1)*1.0/(k+1) for kk in range(k)},
                   **{(-1, kk): 0 for kk in range(k)}}
        self.eplen = 1
        self.starting_states = [0]


class CounterExample1Env(MultiStepEnv):
    """
        infinite-horizon

        0, 0 --> -1: 0
        0, 1 --> 1: 0.9
        1, 0 --> 0: 0.9
        1, 1 --> 1: 0.3

        Theoretically it shouldn't work.
    """
    def __init__(self, absorbing_reward=0, asymmetric_coeff=1, big_reward=0.8):
        super(CounterExample1Env, self).__init__()
        big = big_reward
        small = 0.1

        self.eplen = 10
        normalizing_coeff = 1.0

        big = big/normalizing_coeff  # reward scale
        small = small/normalizing_coeff

        self.states = [0, 1]
        self.actions = [0, 1]
        self.transitions = {
            # -1 is an absorbing state
            (2, 0): 2,
            (2, 1): 2,

            (0, 0): 2,
            (0, 1): 1,
            (1, 0): 0,
            (1, 1): 1,
        }
        self.rewards = {
            (2, 0): absorbing_reward,
            # (2, 1): absorbing_reward,
            (2, 1): asymmetric_coeff*absorbing_reward,  # for asymmetry
            (0, 0): 0,
            (0, 1): big,
            (1, 0): big,
            (1, 1): small
        }
        # Qs are actually not correct
        self.Qs = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 0,

            (2, 0): 0,
            (2, 1): 0,
        }
        # do not specify self.Q
        self.starting_states = [0]

    def step(self, action):
        next_state, reward, done, _ = MultiStepEnv.step(self, action)
        # make this infinite horizon
        return next_state, reward, False, _


def test_CounterExample1Env():
    action_sequences = {
        (0, 0, 0, 0, 0, 0): [0, 2, 2, 2, 2, 2, 2],
        (0, 1, 1, 1, 1, 1): [0, 2, 2, 2, 2, 2, 2],
        (0, 1, 0, 1, 0, 1): [0, 2, 2, 2, 2, 2, 2],
        (1, 0, 0, 0, 0, 0): [0, 1, 0, 2, 2, 2, 2],
        (1, 0, 1, 0, 1, 0): [0, 1, 0, 1, 0, 1, 0],
        (1, 1, 1, 1, 1, 1): [0, 1, 1, 1, 1, 1, 1],
        }


    for action_sequence in action_sequences:
        env = CounterExample1Env()
        states = []
        rewards = []
        dones = []

        state = env.reset()
        states.append(state)

        for action in action_sequence:
            next_state, reward, done, _ = env.step(action)
            states.append(next_state)
            rewards.append(reward)
            dones.append(done)

        print('Action Sequence: {}'.format(action_sequence))
        print('Expected State Sequence: {}'.format(action_sequences[action_sequence]))
        print('Actual State Sequence: {}'.format(action_sequences[action_sequence]))
        print('Rewards: {}'.format(rewards))
        print('Dones: {}'.format(dones))





if __name__ == '__main__':
    # import pprint
    # a = OneStateOneStepKActionEnv(9)
    # print('rewards', a.rewards)
    # print('Qs', a.Qs)
    # print('actions', a.actions)
    # print('states', a.states)
    # print('transitions',a.transitions)

    # print('*'*80)
    # b = OneStateOneStepNineActionEnv()
    # print('rewards', b.rewards)
    # print('Qs', b.Qs)
    # print('actions', b.actions)
    # print('states', b.states)
    # print('transitions',b.transitions)


    # grid_states = create_grid_states(2)
    # print(grid_states)

    # for i in range(8):
    #     print(np.where((grid_states==i)))
    # print(np.where((grid_states==-1)))

    # GW3 = GridWorldK(4)

    test_CounterExample1Env()





