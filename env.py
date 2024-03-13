import numpy as np
class Env:
    def __init__(self):
        self.T = 1000
        self.t = -1
        self.state = np.zeros(2)
        self.action = np.zeros(2)

        # uncomment one of the scenarios
        # scenario a
        self.adv_pert = 0.5 * np.ones((2, self.T))        # out of max 1, each column is the vector w_t
        self.adv_co = 0.5 * np.ones((2, self.T))          # out of max 1, each column is the vector \theta_t^(1)

        # scenario b
        self.adv_pert = 1 * np.ones((2, self.T))  # out of max 1, each column is the vector w_t
        self.adv_co = 1 * np.ones((2, self.T))  # out of max 10, each column is the vector \theta_t^(1)
        for i in range(0, 21, 2):
            self.adv_co[:, i * 50: (i + 1) * 50] *= -0.5

        # scenario c
        self.adv_pert = 0.1 * np.ones((2, self.T))  # out of max 1, each column is the vector w_t
        self.adv_co = 0.1 * np.ones((2, self.T))  # out of max 10, each column is the vector \theta_t^(1)
        for i in range(0, 21, 2):
            self.adv_co[:, i * 50: (i + 1) * 50] *= -5

    def step(self, action):  # needs t to use w_t
        self.t += 1 #first value is 0
        # 1- calculating the reward
        c_t = np.dot(self.adv_co[:, self.t], self.state) # here t is actually starting from 0 (not from 1 as in main).
        # 2- moving
        self.state = 0.9 * self.state + action + self.adv_pert[:, self.t]
        return c_t.item(), self.adv_co[:, self.t], self.state
