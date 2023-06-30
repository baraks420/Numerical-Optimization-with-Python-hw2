import numpy as np


class Function1:
    def __init__(self):
        self.ineq_constraints = [self.ineq1,
                                 self.ineq2,
                                 self.ineq3]
        self.eq_constraints_matrix = np.array([1, 1, 1]).reshape(1, 3)
        self.eq_constraints_res = 1

    def func(self, x):
        f = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 2 * x[2] + 1
        g = np.array([2 * x[0], 2 * (x[1]), 2 * x[2] + 2])
        h = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
        ])
        return f, g, h

    def ineq1(self, x):
        f = -x[0]
        g = np.array([-1, 0, 0])
        h = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        return f, g, h

    def ineq2(self, x):
        f = -x[1]
        g = np.array([0, -1, 0])
        h = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        return f, g, h

    def ineq3(self, x):
        f = -x[2]
        g = np.array([0, 0, -1])
        h = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        return f, g, h


class Function2:
    def __init__(self):
        self.ineq_constraints = [self.ineq1,
                                 self.ineq2,
                                 self.ineq3,
                                 self.ineq4]
        self.eq_constraints_matrix = None
        self.eq_constraints_res = None

    def func(self, x):
        f = -x[0] - x[1]
        g = np.array([-1, -1])
        h = np.array([
            [0, 0],
            [0, 0],
        ])
        return f, g, h

    def ineq1(self, x):
        f = -x[0] - x[1] + 1
        g = np.array([-1, -1])
        h = np.array([
            [0, 0],
            [0, 0],
        ])
        return f, g, h

    def ineq2(self, x):
        f = x[0] - 2
        g = np.array([1, 0])
        h = np.array([
            [0, 0],
            [0, 0],
        ])
        return f, g, h

    def ineq3(self, x):
        f = x[1] - 1
        g = np.array([0, 1])
        h = np.array([
            [0, 0],
            [0, 0],
        ])
        return f, g, h

    def ineq4(self, x):
        f = -x[1]
        g = np.array([0, -1])
        h = np.array([
            [0, 0],
            [0, 0],
        ])
        return f, g, h


class Phi:
    def __init__(self, ineq):
        self.ineq = ineq

    def calc_phi(self, x):
        phi_f = 0
        phi_g = 0
        phi_h1 = 0
        phi_h2 = 0
        try:
            for f in self.ineq:
                f_x, g_x, h_x = f(x)
                phi_f += np.log(-f_x)
                phi_g += g_x / (-f_x)
                phi_h1 += np.dot(g_x, g_x.T) / (f_x ** 2)
                phi_h2 += h_x / (-f_x)
            return -phi_f, phi_g, phi_h1 + phi_h2
        except:
            return None, None, None
