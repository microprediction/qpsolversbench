import pandas as pd
from qpsolvers import available_solvers, solve_qp
from qpsolversbench.problems.financial_problems import Problem
import numpy as np
import scipy

# Worst case discrepancies
# See examples/fin_discrepancy_corr.py

def largest_discrepancy(problem:Problem, solvers_to_not_use=None, with_info=False):
    """ Utility to show if one or more solvers differ from "consensus"
    :param problem
    :param solvers_to_not_use:  [str]
    :param with_info If true returns additional information
    :return:  [ float ]  The solution values for the coordinate where solvers differ the most
    """
    if solvers_to_not_use is None:
        solvers_to_not_use = ['scs','quadprog']

    P, q, G, h, A, b, lb, ub = problem.unpack()
    solvers = [ slv for slv in available_solvers if slv not in solvers_to_not_use ]
    xs = [ solve_qp(P, q, G, h, A, b, solver=slv) for slv in solvers ]
    stacked_solutions = np.column_stack(xs)
    abs_devo = scipy.stats.median_abs_deviation( stacked_solutions, axis=1)
    ndx_worst = np.argmax(abs_devo)
    worst_x = [ x[ndx_worst] for x in xs ]
    info = {'solvers':solvers,
              'ndx_worst':ndx_worst,
              'discrepancy':dict(zip(solvers,worst_x))}
    if with_info:
        return worst_x, info
    else:
        return worst_x


def largest_discrepancy_samples(problem_maker, n_samples:int=5, solvers_to_not_use=None):
    """ Samples of the values taken by x for the coordinate with maximal deviation amongst solvers
    :param problem_maker is a function taking no args and returning a random problem
    :param n_dim:
    :param n_samples:
    :param solvers_to_not_use: [str]
    :return:
    """
    x_disc, report = largest_discrepancy(problem=problem_maker(), solvers_to_not_use=solvers_to_not_use, with_info=True)
    solvers = report['solvers']
    discrep = [largest_discrepancy(problem=problem_maker(), solvers_to_not_use=solvers_to_not_use, with_info=False) for _ in range(n_samples)]
    data = np.array(discrep)
    df = pd.DataFrame(columns=solvers, data=data)
    return df


def largest_discrepancy_corr(problem_maker, n_samples=5, solvers_to_not_use=None):
    """ Correlation between solvers of the coordinate with the maximum absolute deviation
    :param n_dim:     size of P
    :param n_samples:
    :param solvers_to_not_use:
    :return:
    """
    df = largest_discrepancy_samples(problem_maker=problem_maker, n_samples=n_samples, solvers_to_not_use=solvers_to_not_use)
    return df.corr()

