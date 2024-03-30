import numpy as np
import cvxpy as cp


class OptFTRLC():
    def __init__(self, perts, cos, pred_cos, T):
        self.T = T
        self.kappa_M = np.sqrt(40)
        self.cos = cos
        self.perts = perts
        self.pred_cos = pred_cos
        self.M = cp.Variable(40)
        self.M.value = np.zeros(40)

        # projection things
        self.non_projected_sol_parameter = cp.Parameter(40)
        self.objective = cp.Minimize(cp.norm(self.M - self.non_projected_sol_parameter, p=2))
        self.prob = cp.Problem(self.objective, [cp.norm(self.M) <= self.kappa_M])

        self.pred_error = []
        self.acc_error = 0
        self.max_sum = 0

    def get_action(self, t):
        if t < 10:
            available = t
            concatenated_perturbations = np.zeros(20)
            concatenated_perturbations[-2 * available:] = self.perts[:, 0:available].transpose().flatten()
        else:
            concatenated_perturbations = self.perts[:, t - 10:t].transpose().flatten()
        u_1 = np.dot(self.M.value[0: 20], np.flip(concatenated_perturbations))  # 0 to 20 is the first line
        u_2 = np.dot(self.M.value[20: 40], np.flip(concatenated_perturbations))
        return np.array([u_1, u_2]), self.M.value

    '''acc_F_grads is 2 by 40 grad matrix of aggregated until t+1 (what is needed for BTL in the lower timeline'''

    def grad_step(self, t, acc_F_grads):
        # 1- First, adjust the aggregated gradient by subtracting unseen, and adding prediction
        F_adjustment_grad = np.zeros((2, 20))  # 10 2x2 matrices
        F_pred_grad = np.zeros((2, 20))
        for j in range(1, 11):  # parts of the grad
            acc = 0
            acc_pred = 0
            for i in range(0, 11):  # from 0 to 10, total d+1 removals
                for k in range(i, 11):
                    if i == 0 and k == 0:  # the action cost
                        pass
                    else:
                        if t - i + k >= self.T:  # no costs beyond T
                            break
                        else:
                            acc += 0.9 ** k * self.perts[0, t - i - 2] * self.cos[
                                0, t - i + k]  # start from t, always (i.e., t+1)
                            acc_pred += 0.9 ** k * self.perts[0, t - i - 2] * self.pred_cos[0, t - i + k]
            F_adjustment_grad[:,
            (j - 1) * 2: j * 2] = acc  # 0 is the element of the vector \theta_t, both 0 and 1 are same.
            F_pred_grad[:, (j - 1) * 2: j * 2] = acc_pred
        agg_witnessed_F_grad = acc_F_grads - F_adjustment_grad + F_pred_grad
        self.pred_error.append(np.linalg.norm((F_adjustment_grad - F_pred_grad).flatten(), 2))

        # 2- this point onward, use agg_witnessed_F_grad
        lambdaa = self.calc_lambda()
        unprojected_M = (- agg_witnessed_F_grad.flatten()) / max(lambdaa, 0.1)  # to avoid division by 0
        # projection
        self.non_projected_sol_parameter.value = unprojected_M
        self.prob.solve(warm_start=True, solver=cp.ECOS)

    def calc_lambda(self):
        """First access is to self.pred_error[-12] (while predicting M_{t+1}, last full feedback is t-d, i.e., d+1 steps behind)"""
        # most recent access is to [-1-10-1]. (i.e., -1 and -d-1).

        # 1- calculate maximum error of window length 10
        if np.sum(self.pred_error[
                  -22:-12]) > self.max_sum:  # -12 not counted. This exp is the same as "the 10 before the last observed"
            self.max_sum = np.sum(self.pred_error[-22:-12])

        # 2- accumulate [-12] in the root
        if len(self.pred_error) > 12:
            self.acc_error += (self.pred_error[-12]) ** 2
        return (4 * self.max_sum) / (self.kappa_M) + (1 / self.kappa_M) * (np.sqrt(5 * self.acc_error))
