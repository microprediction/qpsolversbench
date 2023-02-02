from qpsolversbench.diagnostics.performance import relative_performance
from qpsolversbench.problems.financial_problems import get_fin_problem
from pprint import pprint

# Example output includes runtimes relative to cvxopt
#
#  {'cvxopt': 1.7413792610168457,
#  'ecos': 17.874003887176514,
#  'ecos_relative': 10.264279750718591,
#  'osqp': 1.917612075805664,
#  'osqp_relative': 1.1012030054187683,
#  'proxqp': 0.74835205078125,
#  'proxqp_relative': 0.42974673440423533}


if __name__=='__main__':
    problem = get_fin_problem(n=2500)
    perf = relative_performance(problem=problem, benchmark_solver='cvxopt')
    pprint(perf)