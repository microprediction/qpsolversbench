#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2022 Stéphane Caron and the qpsolvers contributors.
#
# This file is part of qpsolvers.
#
# qpsolvers is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# qpsolvers is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with qpsolvers. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import scipy.sparse as spa
from qpsolvers.problem import Problem
from precise.skaters.covarianceutil.covrandom import random_factor_cov
from precise.skaters.covarianceutil.covfunctions import nearest_pos_def


def get_fin_problem(n_dim=5) -> Problem:
    """
       print("    min. 1/2 x^T P x + q^T x")
       print("    s.t. G * x <= h")
       print("         A * x == b")
    """
    P = random_factor_cov(n_dim=n_dim)
    P = nearest_pos_def(P)
    q = 0.01 * np.ones(shape=(n_dim, 1)).flatten()
    G = np.eye(n_dim)
    h = np.ones(shape=(n_dim,))
    A = np.ones(shape=(n_dim,))
    b = np.ones(shape=(1,))
    return Problem(P, q, G, h, A, b)


def fin_objective(x, P, q):
    xTP = np.matmul(np.transpose(x), P)
    return np.linalg.norm( np.dot(0.5 * xTP + q, x))


def get_sd3310_problem() -> Problem:
    """
    Get a small dense problem with 3 optimization variables, 3 inequality
    constraints, 1 equality constraint and 0 box constraint.
    """
    M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
    P = np.dot(M.T, M)  # this is a positive definite matrix
    q = np.dot(np.array([3.0, 2.0, 3.0]), M).reshape((3,))
    G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
    h = np.array([3.0, 2.0, -2.0]).reshape((3,))
    A = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0])
    return Problem(P, q, G, h, A, b)


def get_sparse_least_squares(n=150_000):
    """
    Get a sparse least squares problem.

    Parameters
    ----------
    n :
        Problem size.

    Notes
    -----
    This problem was inspired by `this question on Stack Overflow
    <https://stackoverflow.com/q/73656257/3721564>`__.
    """
    # minimize 1/2 || x - s ||^2
    R = spa.eye(n, format="csc")
    s = np.array(range(n), dtype=float)

    # such that G * x <= h
    G = spa.diags(
        diagonals=[
            [1.0 if i % 2 == 0 else 0.0 for i in range(n)],
            [1.0 if i % 3 == 0 else 0.0 for i in range(n - 1)],
            [1.0 if i % 5 == 0 else 0.0 for i in range(n - 1)],
        ],
        offsets=[0, 1, -1],
        format="csc",
    )
    h = np.ones(G.shape[0])

    # such that sum(x) == 42
    A = spa.csc_matrix(np.ones((1, n)))
    b = np.array([42.0]).reshape((1,))

    # such that x >= 0
    lb = np.zeros(n)
    ub = None

    return R, s, G, h, A, b, lb, ub


def get_qpmad_demo_problem():
    """
    Problem from qpmad's `demo.cpp
    <https://github.com/asherikov/qpmad/blob/5e4038f15d85a2a396bb062599f9d7a06d0b0764/test/dependency/demo.cpp>`__.
    """
    P = np.eye(20)
    q = np.ones((20,))
    G = np.vstack([np.ones((1, 20)), -np.ones((1, 20))])
    h = np.hstack([1.5, 1.5])
    lb = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
        ]
    )
    ub = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
    )
    return Problem(P, q, G, h, lb=lb, ub=ub)
