import numpy as np
import cvxpy as cp


class GPC():
    def __init__(self, T, p, g, perts):
        self.kappa_M = np.sqrt(40)  # I will deal with M here as a vector (concatenate two rows)
        self.pert = perts
        self.M = cp.Variable(40)
        self.M.value = np.zeros(40)
        self.t = -1

        self.L = 30     # 20\sqrt{2} coefficient of one M in the history function. equation (8) in the paper
        self.g = 300    # d = p = 10

        # OGD stuff, it knows T
        self.eta = (2 * self.kappa_M) / np.sqrt(self.g * (self.g + self.L * 10 ** 2) * T)  # from agarwal appendix (GPC paper).

        # projection stuff
        self.non_projected_sol_parameter = cp.Parameter(40)
        self.objective = cp.Minimize(cp.norm(self.M - self.non_projected_sol_parameter, p=2))
        self.prob = cp.Problem(self.objective, [cp.norm(self.M) <= self.kappa_M])

    def get_action(self, t):
        # concatenate last 10 perturbation vectors, the concatenated vector should be 2x10 = 20
        self.t = t
        if t < 10:
            available = t
            concatenated_perturbations = np.zeros(20)
            concatenated_perturbations[-2 * available:] = self.pert[:, 0:available].transpose().flatten()
        else:
            concatenated_perturbations = self.pert[:,
                                         t - 10:t].transpose().flatten()  # t is the true time, it is excluded.
        # reversing concatenated_perturbations is to get w_{t-1} first. Then, w_{t-2}, w_{t-3}....
        u_1 = np.dot(self.M.value[0: 20], np.flip(concatenated_perturbations))
        u_2 = np.dot(self.M.value[20: 40], np.flip(concatenated_perturbations))
        return np.array([u_1, u_2]), self.M.value

    def grad_step(self, grad):  # grad,  The grad that will be received here is the flattened
        # update the params based on the revealed grad
        unprojected_M = self.M.value - self.eta * grad
        self.non_projected_sol_parameter.value = unprojected_M
        self.prob.solve(warm_start=True, solver=cp.ECOS)
        # print("Reached GPC Params: ", np.round(self.M.value,decimals=1))
