import numpy as np
from qpsolvers.problem import Problem
from precise.skaters.covarianceutil.covrandom import random_factor_cov
from precise.skaters.covarianceutil.covfunctions import nearest_pos_def


def get_fin_problem(n=5) -> Problem:
    """
       print("    min. 1/2 x^T P x + q^T x")
       print("    s.t. G * x <= h")
       print("         A * x == b")
    """
    P = random_factor_cov(n_dim=n)
    P = nearest_pos_def(P)
    q = 0.01 * np.ones(shape=(n, 1)).flatten()
    G = np.eye(n)
    h = np.ones(shape=(n,))
    A = np.ones(shape=(n,))
    b = np.ones(shape=(1,))
    return Problem(P, q, G, h, A, b)


def fin_objective(x, P, q):
    xTP = np.matmul(np.transpose(x), P)
    return np.linalg.norm( np.dot(0.5 * xTP + q, x))
