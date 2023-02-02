from qpsolvers import print_matrix_vector
from qpsolvers.problem import Problem

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

