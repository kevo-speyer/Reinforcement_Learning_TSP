def get_avail_act(state, n_dest):
    avail_actions = [ dest for dest in range(n_dest) if dest not in state]
    return avail_actions
