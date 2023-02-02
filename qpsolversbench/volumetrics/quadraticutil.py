import numpy as np
from qpsolvers import print_matrix_vector
from qpsolversbench.problems import Problem, get_sd3310_problem
from pprint import pprint

# Minor utilities


def explain_quadratic(problem:Problem):
    P, q, G, h, A, b, lb, ub = problem.unpack()
    print("")
    print("    min. 1/2 x^T P x + q^T x")
    print("    s.t. G * x <= h")
    print("         A * x == b")
    print("")
    print_matrix_vector(P, "P", q, "q")
    print("")
    print_matrix_vector(G, "G", h, "h")
    print("")
    print_matrix_vector(A, "A", b, "b")
    print("")


def check_against_dim3_example(problem):
    """ Show dimensions of inputs and compare to  """
    P, q, G, h, A, b, lb, ub = problem.unpack()
    if np.shape(P)[0]==3:
        problem2 = get_sd3310_problem()
        P2, q2, G2, h2, A2, b2, lb2, ub2 = problem2.unpack()
        dim_match = {'P': np.shape(P) == np.shape(P2),
                     'q': np.shape(q) == np.shape(q),
                     'G': np.shape(G) == np.shape(G2),
                     'h': np.shape(h) == np.shape(h2),
                     'A': np.shape(A) == np.shape(A2),
                     'b': np.shape(b) == np.shape(b2),
                     'lb': np.shape(lb) == np.shape(lb2),
                     'ub': np.shape(ub) == np.shape(ub2),
                     }
        pprint(dim_match)
