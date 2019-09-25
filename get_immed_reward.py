def get_immed_reward(state, action, dist_mat):
    rew = 1./ dist_mat[state[-1],action]
    return rew