from create_dist_matrix import create_dist_matrix, create_dist_mat_2
from get_avail_act import get_avail_act
from get_best_action import get_best_action
from get_or_tools_sol import or_solution
from get_random_traj import get_random_traj
from train import train_model
import numpy as np
#Definitions
n_dest = 20 # Set number of destinations
dist_mat = create_dist_matrix(n_dim = n_dest, opt = 2) # Create distance matrix, opt = 0 is random, opt = 2 is fixed example
#dist_mat = create_dist_mat_2() # Use googles example

def tsp_solver(dist_mat,alpha=0.2,gamma=0.8):
    #alpha is the learning rate
    #gamma is the discount factor
    n_dest = dist_mat.shape[0]
    # Train RL model
    q = train_model(dist_mat, n_train = 2000, gamma = gamma, alpha = alpha)# Get trained transition matrix

    #print(q)

    # Use model to find optimum trajectory
    state = [0]
    distance_travel = 0.
    posible_actions = get_avail_act(state, n_dest)
    while posible_actions: # until all destinations are visited
        action = get_best_action(state[-1], posible_actions, q)
        distance_travel += dist_mat[state[-1], action]
        state.append(action)
        posible_actions = get_avail_act(state, n_dest)

    #Back to warehouse
    action = 0
    distance_travel += dist_mat[state[-1], action]
    state.append(action)

    # Get Best optimization possible
    #print("\nGoogle Results: ")
    best_dist, google_route = or_solution(dist_mat)

    # Get random tour
    random_dist, random_route = get_random_traj(dist_mat)

    #Out RL results
    traj =' -> '.join([str(b) for b in state])
    #print(f"Best trajectory found with RL: \n {traj}" )
    #print(f"Total distance travelled with this traj: {distance_travel}\n")
    slow_pctg = 100*(-1+distance_travel/best_dist)
    random_pctg = 100*(-1+distance_travel/random_dist)
    return slow_pctg, traj, distance_travel, google_route, best_dist
    #print(f"RL solution is {100*(-1+distance_travel/best_dist)}% slower than google's solution")

best_pctg = 100
alpha = 0.012
gamma = 0.4
#for alpha in np.linspace(0.012,0.012,1):
#    for gamma in np.linspace(0.4 ,0.4,100):
for _ in range(15):
        slow_pctg, rl_route, rl_dist, google_route, google_dist = tsp_solver(dist_mat, alpha=alpha, gamma=gamma)
        if slow_pctg < best_pctg:
            best_pctg = slow_pctg
            if slow_pctg < 0:
                print(f"\nBest solution so far with parameters alpha:{alpha}, gamma:{gamma}, is {-np.around(slow_pctg,decimals=1)}% FASTER than google's solution")
                print(f"RL route:     {rl_route}; distance: {rl_dist}")
                print(f"Google route {google_route}; distance: {google_dist}\n")
            else:
                print(f"\nBest solution so far with parameters alpha:{alpha}, gamma:{gamma}, is {np.around(slow_pctg,decimals=1)}% slower than google's solution")
