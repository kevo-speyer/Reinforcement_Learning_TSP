def get_immed_reward(state, action, dist_mat):
    rew = - dist_mat[state[-1],action]
    return rew