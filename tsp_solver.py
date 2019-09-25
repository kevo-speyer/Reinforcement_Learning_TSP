from create_dist_matrix import create_dist_matrix, create_dist_mat_2
from get_avail_act import get_avail_act
from get_best_action import get_best_action
from get_or_tools_sol import or_solution
from train import train_model
#Definitions
n_dest = 6 # Set number of destinations
#dist_mat = create_dist_matrix(n_dim = n_dest) # Create distance matrix
dist_mat = create_dist_mat_2()
n_dest = dist_mat.shape[0]
# Train RL model
q = train_model(dist_mat, n_train = 1000, gamma = 0.2, alpha = 0.1)# Get trained transition matrix

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
print("\nGoogle Results: ")
best_dist = or_solution(dist_mat)

#Out RL results
traj =' -> '.join([str(b) for b in state])
print(f"Best trajectory found with RL: \n {traj}" )
print(f"Total distance travelled with this traj: {distance_travel}\n")

print(f"RL solution is {100*(-1+distance_travel/best_dist)}% slower than google's solution")
