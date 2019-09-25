def get_best_action(curr_state, posible_actions, q):
    best_action_index = q[curr_state, posible_actions].argmax()

    return posible_actions[best_action_index]