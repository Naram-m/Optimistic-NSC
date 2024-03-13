import numpy as np
from env import Env
from gpc import GPC
from optimal import Optimal
from OptFTRLC import OptFTRLC

np.random.seed(1994)

'''this is the max norm of a vector of length p * size_M, each element can reach up to 10 * 1/delta.
it is the derivative of the cost funciton, which is just [-1 to 1] x.'''
g = np.linalg.norm(np.ones(10 * 2 * 2) * 1 * (1 / 0.1), 2)  # here it is delta (small),
print("Max g: ", g)

env1 = Env()
env3 = Env()
env4 = Env()

perts = env1.adv_pert
cos = env1.adv_co
T = env1.T

##### Forming the predictions  #####
good_pred_cos = np.zeros((2, T))
for co_index in range(T):
    if np.random.random() <= 0.9:
        good_pred_cos[:, co_index] = cos[:, co_index]
    else:
        good_pred_cos[:, co_index] = np.random.uniform(-1, 1)
bad_pred_cos = np.zeros((2, T))
for co_index in range(T):
    if np.random.random() <= 0.1:
        bad_pred_cos[:, co_index] = cos[:, co_index]
    else:
        bad_pred_cos[:, co_index] = np.random.uniform(-1, 1)


##### Calculating Grads of c_t(x_t)  #####
grads = []
for t in range(1, T + 1):
    grad = np.zeros((2, 20))    # 10 2x2 matrices
    for j in range(1, 11):      # 10, for each M
        acc = 0
        for i in range(20):     # 20 is how far to go back in time, supposedly beginning but 20 is already 1e-5 approx.
            if t - j - i - 1 < 0:
                break
            else:
                acc += 0.9 ** i * perts[0, t - j - i - 2]  # 0 is the element of the vector w_t, both 0 and 1 are the same.
        # fill the 4 elements with the same grad value (since w1 and w2 are the same, and \theta1 and \theta2 are the same.)
        grad[:, (j - 1) * 2: j * 2] = cos[0, t - 1] * acc  # 0 is the element of the vector \theta_t, both 0 and 1 are same.
    grads.append(grad)  # grads is a list, each element is a full 2x20 gradient
acc_grads = np.cumsum(grads, axis=0) #0 verified, see tests
#######################################

##### Calculating F Grads #####
F_grads = []
for t in range(1, T + 1):
    F_grad = np.zeros((2, 20))     # 10 2x2 matrices
    for j in range(1, 11):         # 10, for each M
        acc = 0
        for i in range(1, 11): # this is the (i) on f subscript, 0 is skipped because no action cost.
            if t - j - 2 < 0 or t + i > T:  # no negative indexing, or costs beyond T
                break
            else:
                acc += 0.9 ** i * perts[0, t - j - 2] * cos[0, t - 1 + i]
        # fill the 4 elements with the same grad value
        F_grad[:, (j - 1) * 2: j * 2] = acc  # 0 is the element of the vector \theta_t, both 0 and 1 are same.
    F_grads.append(F_grad)  # grads is a list, each element is a full 2x20 gradient
acc_F_grads = np.cumsum(F_grads, axis=0)
#######################################
gpc1 = GPC(T, 10, g, perts)
gpc1_costs = []

optimal1 = Optimal(T, 10, g, perts=perts)
optimal_costs = []
############## The Interaction Loop ################
_1 = 0
for t in range(1, T + 1):
    if t % 500 == 0:
        print("t: ", t)
        print("Reached GPC Params: ", np.round(_1, decimals=2))

    # 1- ask each agent for an action
    u_gpc, _1 = gpc1.get_action(t)
    u_optimal, _2 = optimal1.get_action(t, grads[t-1].flatten())

    # 2- report it back to env and ask the env about the cost and the new state
    (cost, coe, state) = env1.step(u_gpc)

    # 3- Form the gradient, a vector of length 10
    grad = grads[t - 1] # grads start from 0,

    # 4- send the grad to the GPC agent
    gpc1.grad_step(grad.flatten())

    # 5- record the cost
    gpc1_costs.append(cost)

    # appendix: counter factual accumelated cost of the optimal
    env2 = Env()
    acc_cost2 = 0
    for sub_t in range (1, t+1):
        (cost2, coe2, state2) = env2.step(u_optimal)
        acc_cost2 += cost2
    optimal_costs.append(acc_cost2)

############## The OptFTRL-C Interaction Loop ################
_3 = 0
opt_ftrlc1 = OptFTRLC(perts, cos, good_pred_cos, T)
opt_ftrlc1_costs = []
for t in range(1, T + 1):
    if t % 500 == 0:
        print("OptFTRL-C, t: ", t)
        print("Reached Optimistic Params: ", np.round(_3, decimals=2))
    u_OptFTRL_C, _3 = opt_ftrlc1.get_action(t)
    (cost3, coe3, state3) = env3.step(u_OptFTRL_C)
    acc_F_grad = acc_F_grads[t - 1] # accumulated until t
    opt_ftrlc1.grad_step(t, acc_F_grad)
    opt_ftrlc1_costs.append(cost3)

_4 = 0
opt_ftrlc1_bad = OptFTRLC(perts, cos, bad_pred_cos, T)
opt_ftrlc1_bad_costs = []
for t in range(1, T + 1):
    if t % 500 == 0:
        # print("OptFTRL-C, t: ", t)
        # print("Reached Optimistic Params: ", np.round(_3, decimals=2))
        pass
    u_OptFTRL_C, _4 = opt_ftrlc1_bad.get_action(t)
    (cost4, coe4, state4) = env4.step(u_OptFTRL_C)
    acc_F_grad = acc_F_grads[t - 1] # accumulated until t
    opt_ftrlc1_bad.grad_step(t, acc_F_grad)
    opt_ftrlc1_bad_costs.append(cost4)

####################### Writing results ##############################

with open('./results/gpc.npy', 'wb') as f:
    np.save(f, np.array(gpc1_costs, dtype=object))

with open('./results/optftrlc_good.npy', 'wb') as f:
    np.save(f, np.array(opt_ftrlc1_costs, dtype=object))

with open('./results/optftrlc_bad.npy', 'wb') as f:
    np.save(f, np.array(opt_ftrlc1_bad_costs, dtype=object))

with open('./results/optimal.npy', 'wb') as f:
    np.save(f, np.array(optimal_costs, dtype=object))
