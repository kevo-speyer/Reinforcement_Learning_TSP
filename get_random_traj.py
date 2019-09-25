from numpy import random
def get_random_traj(dist_mat):
    n_dest = dist_mat.shape[0]
    route = [0]
    avail_destinations = list(range(1,n_dest))
    dist_trav = 0.
    while avail_destinations:
        new_dest = random.choice(avail_destinations)
        avail_destinations.remove(new_dest)
        dist_trav += dist_mat[route[-1],new_dest]
        route.append(new_dest)

    return dist_trav, route
