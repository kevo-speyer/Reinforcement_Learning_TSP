import numpy as np
from create_dist_matrix import create_dist_matrix
from get_avail_act import get_avail_act
from get_next_action import get_next_action
from update_q import update_q

def train_model(dist_mat, n_train = 1000, gamma = 0.8, alpha = 0.5):
    #n_train = 1000 # number of training trips
    #gamma = 0.8 # Bellman's Parameter for Reinforcement Learning
    #dist_mat = create_dist_matrix(n_dim = n_dest)
    n_dest =  dist_mat.shape[0] # number of cities to visit
    q = np.zeros([n_dest,n_dest]) # transition matrix to train
    epsilon = 1.
    for i in range(n_train):
        state = [0] # initial state = wharehouse
        posible_actions = get_avail_act(state, n_dest)

        while posible_actions: # until all destinations are visited
            action = get_next_action(q, posible_actions, state, epsilon) # np.random.choice(posible_actions)
            q = update_q(q, dist_mat, gamma, state, action, alpha)
            state.append(action)
            posible_actions = get_avail_act(state, n_dest)

        # Last trip: from last destination to wharehouse
        action = 0
        q = update_q(q, dist_mat, gamma, state, action, alpha)
        state.append(0)
        epsilon = 1. - i * 1/n_train

    return q



def train_model_debug(dist_mat, n_train = 1000, gamma = 0.8, alpha = 0.5):
    """Same as train_model, bu also returns total distance vs iteration number"""
    #n_train = 1000 # number of training trips
    #gamma = 0.8 # Bellman's Parameter for Reinforcement Learning
    #dist_mat = create_dist_matrix(n_dim = n_dest)
    n_dest =  dist_mat.shape[0] # number of cities to visit
    q = np.zeros([n_dest,n_dest]) # transition matrix to train
    epsilon = 1.
    distances = []
    for i in range(n_train):
        state = [0] # initial state = wharehouse
        posible_actions = get_avail_act(state, n_dest)
        distance_traveled = 0

        while posible_actions: # until all destinations are visited
            action = get_next_action(q, posible_actions, state, epsilon) # np.random.choice(posible_actions)
            distance_traveled += dist_mat[state[-1], action]
            q = update_q(q, dist_mat, gamma, state, action, alpha)
            state.append(action)
            posible_actions = get_avail_act(state, n_dest)

        # Last trip: from last destination to wharehouse
        action = 0
        distance_traveled += dist_mat[state[-1], action]
        q = update_q(q, dist_mat, gamma, state, action, alpha)
        state.append(0)
        epsilon = 1. - i * 1/n_train
        distances.append(distance_traveled)

    return q, distances
