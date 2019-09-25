from numpy import random

from get_best_action import get_best_action


def get_next_action(q, possible_actions, state, epsilon):
    if random.random() < epsilon:
        action = random.choice(possible_actions)
    else:
        action = get_best_action(state[-1], possible_actions, q)
    return action