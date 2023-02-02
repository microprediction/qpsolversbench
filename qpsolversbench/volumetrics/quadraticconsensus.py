import pandas as pd
from qpsolvers import available_solvers, solve_qp
from qpsolversbench.problems import get_fin_problem, fin_objective
from pprint import pprint
import numpy as np
import scipy

# An examination of discrepancies in the solutions of quadratic problems


def largest_discrepancy(n_dim=2, solvers_to_not_use=None, with_report=False):
    """ Utility to show if one or more solvers differ from "consensus"
    :param n_dim:
    :param solvers_to_not_use:  [str]
    :return:
    """
    if solvers_to_not_use is None:
        solvers_to_not_use = ['scs','quadprog']  # Remove ecos after patch

    problem = get_fin_problem(n_dim=n_dim)
    P, q, G, h, A, b, lb, ub = problem.unpack()
    solvers = [ slv for slv in available_solvers if slv not in solvers_to_not_use ]
    xs = [ solve_qp(P, q, G, h, A, b, solver=slv) for slv in solvers ]
    solutions = dict(zip(solvers,xs))
    stacked_solutions = np.column_stack(xs)
    abs_devo = scipy.stats.median_abs_deviation( stacked_solutions, axis=1)
    ndx_worst = np.argmax(abs_devo)
    worst_x = [ x[ndx_worst] for x in xs ]
    report = {'solvers':solvers,
              'ndx_worst':ndx_worst,
              'discrepancy':dict(zip(solvers,worst_x))}
    if with_report:
        return worst_x, report
    else:
        return worst_x


def solver_discrepancy_samples(n_dim=2, n_samples=5, solvers_to_not_use=None):
    """ Samples of the values taken by x for the coordinate with maximal deviation amongst solvers
    :param n_dim:
    :param n_samples:
    :param solvers_to_not_use:
    :return:
    """
    x_disc, report = largest_discrepancy(n_dim=n_dim, solvers_to_not_use=solvers_to_not_use, with_report=True)
    solvers = report['solvers']
    discrep = [largest_discrepancy(n_dim=n_dim, solvers_to_not_use=solvers_to_not_use, with_report=False) for _ in range(n_samples)]
    data = np.array(discrep)
    print(np.shape(data))
    df = pd.DataFrame(columns=solvers, data=data)
    return df


def solver_discrepancy_corrcoef(n_dim=2, n_samples=5, solvers_to_not_use=None):
    """ Correlation between solvers of the coordinate with the maximum absolute deviation
    :param n_dim:     size of P
    :param n_samples:
    :param solvers_to_not_use:
    :return:
    """
    df = solver_discrepancy_samples(n_dim=n_dim, n_samples=n_samples, solvers_to_not_use=solvers_to_not_use)
    return df.corr()


if __name__=='__main__':
    corr_mat = solver_discrepancy_corrcoef(n_dim=500, n_samples=200)
    pprint(corr_mat)