from get_immed_reward import get_immed_reward


def update_q(q, dist_mat, gamma, state, action, alpha):
    imm_reward = get_immed_reward(state,action,dist_mat)
    delayed_reward = q[action,:].max()
    q[state[-1],action] += alpha * (imm_reward + gamma * delayed_reward - q[state[-1],action])
    return q