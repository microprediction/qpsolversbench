from qpsolversbench.problems.financial_problems import get_fin_problem
from qpsolversbench.diagnostics.largest_discrepancy import largest_discrepancy_corr
from pprint import pprint

# This example looks at the correlation matrix of the worst discrepancy between optimizers
# It will just be NaN if the solvers always agree, but if not and example output might be:
#
#            cvxopt      ecos      osqp    proxqp
# cvxopt  1.000000  1.000000  0.748112  0.950912
# ecos    1.000000  1.000000  0.748178  0.950950
# osqp    0.748112  0.748178  1.000000  0.713512
# proxqp  0.950912  0.950950  0.713512  1.000000
#
# Here we observe that cvxopt and ecos agree (both interior point btw) and
# proxqp is in some sense a bit closer to them than osqp. Interpret with care!

if __name__ == '__main__':
    def fin_problem_maker():
        return get_fin_problem(n=100)
    corr_mat = largest_discrepancy_corr(problem_maker=fin_problem_maker, n_samples=100)
    pprint(corr_mat)
