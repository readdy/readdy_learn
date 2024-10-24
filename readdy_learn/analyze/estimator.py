# coding=utf-8

# Copyright © 2016 Computational Molecular Biology Group,
#                  Freie Universität Berlin (GER)
#
# This file is part of ReaDDy.
#
# ReaDDy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General
# Public License along with this program. If not, see
# <http://www.gnu.org/licenses/>.

"""
Created on 19.05.17

@author: clonker
"""

import itertools

import numpy as np
import scipy.optimize as so
from pathos.multiprocessing import Pool
from readdy_learn.analyze_tools import opt
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit

import readdy_learn.analyze.derivative as deriv

_epsilon = np.sqrt(np.finfo(float).eps)


class ReaDDyElasticNetEstimator(BaseEstimator):
    def __init__(self, trajs, basis_function_configuration, alpha=1.0, l1_ratio=1.0, init_xi=None,
                 verbose=False, maxiter=15000, approx_jac=True, method='SLSQP', options=None, rescale=True, tol=1e-16,
                 constrained=True):
        """

        :param trajs:
        :param basis_function_configuration:
        :param alpha:
        :param l1_ratio:
        :param init_xi:
        :param verbose:
        :param maxiter:
        :param approx_jac:
        :param method: one of SLSQP, L-BFGS-B, TNC
        """
        self.basis_function_configuration = basis_function_configuration
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.rescale = rescale
        if not isinstance(trajs, (list, tuple)):
            self.trajs = [trajs]
        else:
            self.trajs = trajs
        if init_xi is None:
            self.init_xi = np.array([.5] * self.basis_function_configuration.n_basis_functions)
        else:
            self.init_xi = init_xi
        self.verbose = verbose
        self.maxiter = maxiter
        self.approx_jac = approx_jac
        self.method = method
        self.options = options if options is not None else {}
        self.tol = tol
        self.constrained = constrained

    def _get_slice(self, traj_range):
        if traj_range is not None:
            if isinstance(traj_range, tuple) and len(traj_range) == 2 and len(self.trajs) > 1:
                data = self.trajs[traj_range[0]].counts[traj_range[1]]
                expected = self.trajs[traj_range[0]].dcounts_dt[traj_range[1]]
            else:
                data = self.trajs[0].counts[traj_range]
                expected = self.trajs[0].dcounts_dt[traj_range]
        else:
            data = [np.ascontiguousarray(t.counts) for t in self.trajs]
            expected = [np.ascontiguousarray(t.dcounts_dt) for t in self.trajs]
        return data, expected

    def fit(self, X, y=None):
        """
        sklearn compatibility

        :param X: Usually data matrix. In this context a temporal range of the trajectories, to be fitted.
        :param y: Usually the expected vector. In this context unused.
        """
        return self.fit_trajs(X)

    def get_theta(self, data):
        if not isinstance(data, np.ndarray):
            thetas = []
            for d in data:
                large_theta = np.array([f(d) for f in self.basis_function_configuration.functions])
                large_theta = np.ascontiguousarray(np.transpose(large_theta, axes=(1, 0, 2)))
                thetas.append(large_theta)
            return thetas
        else:
            large_theta = np.array([f(data) for f in self.basis_function_configuration.functions])
            large_theta = np.ascontiguousarray(np.transpose(large_theta, axes=(1, 0, 2)))
            return large_theta

    def get_analytical_jac(self):
        data, expected = self._get_slice(None)
        large_theta = self.get_theta(data)
        return lambda x: opt.elastic_net_objective_fun_jac(x, self.alpha, self.l1_ratio, large_theta, expected)

    def get_approximated_jac(self):
        data, expected = self._get_slice(None)
        large_theta = self.get_theta(data)

        def objective(x):
            obj = opt.elastic_net_objective_fun(x, self.alpha, self.l1_ratio, large_theta, expected)
            # print("got {}".format(obj))
            return obj

        def wrap_function(function, args):
            ncalls = [0]
            if function is None:
                return ncalls, None

            def function_wrapper(*wrapper_args):
                ncalls[0] += 1
                return function(*(wrapper_args + args))

            return function_wrapper

        return wrap_function(deriv.approx_jacobian, (objective, _epsilon))

    def fit_trajs(self, traj_range):
        """
        We hand in the data on construction, thus the fit method only requires the range of the trajectories to be fitted.

        :param traj_range: either a sequence of time indices if self.trajs is a single trajectory,
        or a tuple of trajectory index and time indices if multiple trajectories are available
        :return: self
        """

        data, expected = self._get_slice(traj_range)

        large_theta = self.get_theta(data)

        if self.constrained:
            bounds = [(0., None)] * self.basis_function_configuration.n_basis_functions
        else:
            bounds = None
        init_xi = self.init_xi

        jac = False if self.approx_jac else \
            lambda x: opt.elastic_net_objective_fun_jac(x, self.alpha, self.l1_ratio, large_theta, expected)

        options = {'disp': False, 'maxiter': self.maxiter, 'iprint': 2}
        if self.method == 'L-BFGS-B':
            options['maxfun'] = self.maxiter
        options.update(self.options)

        def objective(x):
            obj = opt.elastic_net_objective_fun(x, self.alpha, self.l1_ratio, large_theta, expected)
            return obj

        result = so.minimize(
            objective,
            init_xi,
            bounds=bounds,
            tol=self.tol,
            method=self.method,
            jac=jac,
            options=options)

        self.coefficients_ = result.x

        self.success_ = result.success
        if self.verbose:
            if not result.success:
                print("optimization problem did not exit successfully (alpha=%s, lambda=%s)!" % (
                    self.alpha, self.l1_ratio))
            else:
                print("optimization problem did exit successfully (alpha=%s, lambda=%s)!" % (self.alpha, self.l1_ratio))
            print("status %s: %s" % (result.status, result.message))
            print("%s / %s iterations" % (result.nit, self.maxiter))
        self.result_ = result
        return self

    def score(self, X, y):
        data, _ = self._get_slice(X)
        large_theta = np.array([f(data) for f in self.basis_function_configuration.functions])
        large_theta = np.transpose(large_theta, axes=(1, 0, 2))
        return -1. * opt.score(self.coefficients_, large_theta, y)


class CrossTrajSplit(object):
    def split(self, range):
        return [(range, None)]


class CV(object):
    def __init__(self, trajs, bfc, alphas, l1_ratios, n_splits, init_xi, n_jobs=8, show_progress=True,
                 mode='k_fold', verbose=False, method='SLSQP', maxiter=300000, rescale=True, tol=1e-16):
        self.alphas = alphas
        self.l1_ratios = l1_ratios
        self.n_jobs = n_jobs
        self.n_splits = n_splits
        self.trajs = trajs
        if not isinstance(self.trajs, (list, tuple)):
            self.trajs = [self.trajs]
        self.bfc = bfc
        self.show_progress = show_progress
        self.result = []
        self.init_xi = init_xi
        self.mode = mode
        self.verbose = verbose
        self.method = method
        self.maxiter = maxiter
        self.rescale = rescale
        self.tol = tol

    def compute_cv_result_cross_trajs(self, params):
        alpha, l1_ratio, (train_ix, test_ix) = params
        scores = []

        train_trajs = [self.trajs[ix] for ix in train_ix]
        test_trajs = [self.trajs[ix] for ix in test_ix]
        estimator = ReaDDyElasticNetEstimator(train_trajs, self.bfc, alpha=alpha, maxiter=self.maxiter,
                                              l1_ratio=l1_ratio, init_xi=self.init_xi, verbose=self.verbose,
                                              method=self.method, rescale=self.rescale, tol=self.tol)
        # fit the whole thing
        estimator.fit(None)
        if estimator.success_:
            testimator = ReaDDyElasticNetEstimator(test_trajs, self.bfc, alpha=alpha,
                                                   l1_ratio=l1_ratio, init_xi=self.init_xi,
                                                   verbose=self.verbose,
                                                   method=self.method, rescale=self.rescale, tol=self.tol)
            testimator.coefficients_ = estimator.coefficients_
            ttraj = testimator.trajs[0]
            score = testimator.score(range(0, ttraj.n_time_steps), ttraj.dcounts_dt)
            scores.append(score)

        else:
            print("no success for alpha={}, l1_ratio={}".format(alpha, l1_ratio))
            print("status %s: %s" % (estimator.result_.status, estimator.result_.message))
            print("%s / %s iterations" % (estimator.result_.nit, self.maxiter))

        return {'scores': scores, 'alpha': alpha, 'l1_ratio': l1_ratio, 'train_trajs': train_ix, 'test_trajs': test_ix}

    def fit_cross_trajs(self):
        from sklearn.model_selection import LeaveOneOut

        loo = LeaveOneOut()
        train_sets = []
        for train, test in loo.split(np.array(range(len(self.trajs)), dtype=int)):
            train_sets.append((train, test))
        params = itertools.product(self.alphas, self.l1_ratios, train_sets)
        result = []
        with Pool(processes=self.n_jobs) as p:
            for idx, res in enumerate(p.imap_unordered(self.compute_cv_result_cross_trajs, params, 1)):
                result.append(res)
        self.result = result

    def compute_cv_result(self, params):
        if self.mode == 'k_fold':
            kf = KFold(n_splits=self.n_splits)
        elif self.mode == 'time_series_split':
            kf = TimeSeriesSplit(n_splits=self.n_splits)
        elif self.mode == 'full_cross_traj':
            kf = CrossTrajSplit()
        else:
            print("unknown mode: %s" % self.mode)
            return
        alpha, l1_ratio = params
        estimator = ReaDDyElasticNetEstimator(self.trajs, self.bfc, alpha=alpha, maxiter=self.maxiter,
                                              l1_ratio=l1_ratio, init_xi=self.init_xi, verbose=self.verbose,
                                              method=self.method, rescale=self.rescale, tol=self.tol)
        if self.test_traj is not None:
            test_estimator = ReaDDyElasticNetEstimator(self.test_traj, self.bfc, alpha=alpha,
                                                       l1_ratio=l1_ratio, init_xi=self.init_xi, verbose=self.verbose,
                                                       method=self.method, rescale=self.rescale, tol=self.tol)
        scores = []
        for train_idx, test_idx in kf.split(range(0, self.trajs.n_time_steps)):
            estimator.fit(train_idx)
            if estimator.result_.success:
                if self.test_traj is not None:
                    test_estimator.coefficients_ = estimator.coefficients_
                    scores.append(
                        test_estimator.score(range(0, self.test_traj.n_time_steps), self.test_traj.dcounts_dt))
                else:
                    scores.append(estimator.score(test_idx, self.trajs.dcounts_dt[test_idx]))
        return {'scores': scores, 'alpha': alpha, 'l1_ratio': l1_ratio}

    def fit(self):
        params = itertools.product(self.alphas, self.l1_ratios)
        result = []
        with Pool(processes=self.n_jobs) as p:
            for idx, res in enumerate(p.imap_unordered(self.compute_cv_result, params, 1)):
                result.append(res)
        self.result = result


def get_dense_params(traj, bfc, n_initial_values=16, n_jobs=8, initial_value=None):
    if initial_value is not None:
        n_initial_values = 1
        initial_values = [initial_value]
    else:
        initial_values = [np.random.random(bfc.n_basis_functions) for _ in range(n_initial_values)]

    def worker(init_xi):
        est = ReaDDyElasticNetEstimator(traj, bfc, alpha=0, l1_ratio=1.0, init_xi=init_xi, verbose=False)
        est.fit(range(0, traj.n_time_steps))
        return est.coefficients_

    coeffs = []
    with Pool(processes=n_jobs) as p:
        for idx, coeff in enumerate(p.imap_unordered(worker, initial_values, 1)):
            coeffs.append(coeff)
            f.value = idx + 1
    f.close()
    return coeffs
