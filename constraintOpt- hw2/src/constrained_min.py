import numpy as np
from scipy.linalg import cho_factor

from tests import examples

epsilon = 10 ** -10


def minimize(func, calc_phi, eq_constraints_mat, eq_constraints_rhs, x, t):
    for i in range(1000):
        phi = calc_phi(x)
        f = func(x)

        barrier = t * f[0] + phi[0]
        barrier_g = t * f[1] + phi[1]
        barrier_h = t * f[2] + phi[2]
        if eq_constraints_mat is None:
            p = calc_newton_direction_unconstrained(barrier_g, barrier_h)
        else:
            p = calc_newton_direction_constraint(eq_constraints_mat, barrier_g, barrier_h)
        l = np.sqrt(np.dot(p, np.dot(barrier_h, p.T)))
        if 0.5 * (l ** 2) < epsilon:
            break
        a = wolfe_condition_with_backtracking(func, t, calc_phi, x, barrier, barrier_g, p)
        x += a * p
    return x


def calc_newton_direction_constraint(eq_constraints_mat, gradient, hessian):
    mat_shape = eq_constraints_mat.shape
    zero = np.zeros((mat_shape[0], mat_shape[0]))

    A = np.block([
        [hessian, eq_constraints_mat.T],
        [eq_constraints_mat, zero],
    ])
    B = np.block([[-gradient, 0]]).T
    p = np.linalg.solve(A, B)[0:mat_shape[1]].reshape((mat_shape[1]))

    return p


def calc_newton_direction_unconstrained(gradient, hessian):
    tau = 10**-6
    while True:
        try:
            h = hessian + tau * np.identity(len(hessian))
            np.linalg.cholesky(h)
            break
        except:
            tau *= 10
    p = - np.dot(np.linalg.inv(h),gradient.reshape(1, len(gradient)).T)
    p = p.reshape(len(p))

    return p


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    t = 1
    mu = 10
    x = x0
    phi = examples.Phi(ineq_constraints).calc_phi
    m = len(ineq_constraints)
    x_list = [x]
    objectives = [func(x)[0]]
    while m / t >= epsilon:
        x = minimize(func, phi, eq_constraints_mat, eq_constraints_rhs, x, t)
        x_list.append(x)
        objectives.append(func(x)[0])
        t *= mu
    return x_list,objectives


def wolfe_condition_with_backtracking(func, t, calc_phi, x, val, gradient, direction, alpha=0.01, beta=0.5):
    step_length = 1
    while True:
        next_f_x = t * func(x + step_length * direction)[0] + calc_phi(x + step_length * direction)[0]
        if next_f_x <= val + alpha * step_length * gradient.dot(direction):
            return step_length
        step_length *= beta

