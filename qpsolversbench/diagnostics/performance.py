from qpsolvers import available_solvers, solve_qp
from qpsolvers.problem import Problem
import time

# A timer
# See /examples/fin_performance.py


def relative_performance(problem:Problem, solvers_to_not_use=None, benchmark_solver='cvxopt'):
    """ Utility to show runtimes
    :param solvers_to_not_use:  [str]
    :param benchmark_solver: str used to determine "1" in relative time scale
    :return:
    """
    if solvers_to_not_use is None:
        solvers_to_not_use = ['scs','quadprog']  # Remove ecos after patch

    cpu = dict()
    P, q, G, h, A, b, lb, ub = problem.unpack()
    solvers_to_use = [ slv for slv in available_solvers if slv not in solvers_to_not_use ]

    for solver in solvers_to_use:
        print(solver)
        st = time.time()
        try:
            _x = solve_qp(P, q, G, h, A, b, solver=solver)
            cpu[solver] = time.time() - st
        except Exception as e:
            cpu[solver] = str(e)

    benchmarked_solvers = [ slv for slv in solvers_to_use if not slv==benchmark_solver]
    for slv in benchmarked_solvers:
        cpu[slv+'_relative'] = cpu[slv]/cpu[benchmark_solver]
    return cpu

