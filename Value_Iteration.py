# Solving 8x8 frozen-lake environment using value iteration

import gym
import numpy as np
import time

env_frozen_lake = gym.make('FrozenLake8x8-v0')
# Solving MDP using exact method needs to access transition function (i.e., model dynamics)
# prob, next_state, reward, done = env.P[state][action][]
# where next_state == row*nCol + col

# Environment
# S: start, G: goal, F: frozen, H: hole
# "SFFFFFFF",
# "FFFFFFFF",
# "FFFHFFFF",
# "FFFFFHFF",
# "FFFHFFFF",
# "FHHFFFHF",
# "FHFFHFHF",
# "FFFHFFFG"

# constants
nRow = 8
nCol = 8
nA = 4
nS = nRow * nCol
EPS = 1e-4


def to_s(num):
    r = int(num / nRow)
    c = int(num % nRow)
    return r, c


def to_matrix(val_table):
    matrix = np.zeros((nRow, nCol))

    for i in range(len(val_table)):
        ro, co = to_s(i)
        matrix[ro][co] = val_table[i]

    return matrix


def print_matrix(val_matrix):
    for i in range(nRow):
        for j in range(nCol):
            print('{:.3f} '.format(val_matrix[i][j]), end=' ')
        print(' ')
    print('-------------------------------------------------------')


def print_policy_matrix(p):
    policy_matrix = to_matrix(p)

    print('-------------------------------------------------------')
    print('optimal policy table')
    # LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
    for i in range(nRow):
        for j in range(nCol):
            if policy_matrix[i][j] == 0:
                print('< ', end=' ')
            elif policy_matrix[i][j] == 1:
                print('v ', end=' ')
            elif policy_matrix[i][j] == 2:
                print('> ', end=' ')
            else:
                print('^ ', end=' ')
        print(' ')
    print('-------------------------------------------------------')


def value_iteration(env, max_iter=10000, gamma=0.9):
    state_values = np.zeros((nS,))
    # LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
    opt_policy = np.zeros((nS,))

    for it in range(max_iter):
        state_values_ = np.copy(state_values)

        for i in range(nS):
            for j in range(nA):
                val = 0
                for k in range(len(env.P[i][j])):
                    prob, next_state, reward, done = env.P[i][j][k]
                    val += prob * (reward + gamma * state_values_[next_state])
                if val > state_values[i]:
                    state_values[i] = val
                    opt_policy[i] = j

        print('value table (iteration {})'.format(it + 1))
        print_matrix(to_matrix(state_values))

        if np.abs(np.sum(state_values) - np.sum(state_values_)) < EPS:
            print('value iteration stop at {}'.format(it + 1))
            return to_matrix(state_values), opt_policy


def rendering(env, p, max_step=300):
    obs = env.reset()
    env.render()

    for i in range(max_step):
        print('{} steps'.format(i + 1))
        obs, rew, finish, _ = env.step(int(p[obs]))
        env.render()
        time.sleep(1.0)

        if finish and rew == 1.0:
            print('goal')
            break

        if finish and rew == 0.0:
            print('hole')
            break


def score(env, p, max_ep=1000):
    holes = 0
    step_list = np.array([])

    for ep in range(max_ep):
        obs = env.reset()

        for step in range(100000):
            obs, rew, finish, _ = env.step(int(p[obs]))

            if finish and rew == 0.0:
                holes += 1
                break

            if finish:
                step_list = np.append(step_list, step + 1)
                break

    print('Took {} steps to get the frisbee'.format(np.mean(step_list)))
    print('{:.1f}% fail to get the frisbee'.format(holes / max_ep * 100.0))


# main
_, policy = value_iteration(env_frozen_lake)
print_policy_matrix(policy)
# rendering(env_frozen_lake, policy)
score(env_frozen_lake, policy)