import numpy as np
import cvxpy as cp


class Optimal():
    def __init__(self, T, p, g, grads=None, perts=None):
        self.pert = perts
        self.kappa_M = np.sqrt(40)
        self.M = cp.Variable(40)
        self.M.value = np.zeros(40)
        self.acc_grads = np.zeros(40)

        self.acc_grads_param = cp.Parameter(40)
        self.objective = cp.Minimize(self.acc_grads_param @ self.M)
        self.prob = cp.Problem(self.objective, [cp.norm(self.M) <= self.kappa_M])

    def get_action(self, t, grad):
        self.acc_grads += grad
        self.acc_grads_param.value = self.acc_grads
        self.prob.solve(warm_start=True, solver=cp.ECOS)
        if t < 10:
            available = t
            concatenated_perturbations = np.zeros(20)
            concatenated_perturbations[-2 * available:] = self.pert[:, 0:available].transpose().flatten()
        else:
            concatenated_perturbations = self.pert[:, t - 10:t].transpose().flatten()
        u_1 = np.dot(self.M.value[0: 20], np.flip(concatenated_perturbations))
        u_2 = np.dot(self.M.value[20: 40], np.flip(concatenated_perturbations))
        return np.array([u_1, u_2]), self.M.value
