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

import analyze_tools.opt as opt
import numpy as np
import scipy.optimize as so
from pathos.multiprocessing import Pool
from sklearn.linear_model.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit


class ConversionReaction(object):
    def __init__(self, type1, type2, n_species):
        self.type1 = type1
        self.type2 = type2
        self.n_species = n_species

    def __call__(self, concentration):
        if len(concentration.shape) == 1:
            concentration = np.expand_dims(concentration, axis=0)
            result = np.zeros((1, self.n_species))
        else:
            result = np.zeros((concentration.shape[0], self.n_species))
        result[:, self.type1] = -concentration[:, self.type1]
        result[:, self.type2] = concentration[:, self.type1]
        return result.squeeze()


class FusionReaction(object):
    def __init__(self, type_from1, type_from2, type_to, n_species):
        self.type_from1 = type_from1
        self.type_from2 = type_from2
        self.type_to = type_to
        self.n_species = n_species

    def __call__(self, concentration):
        if len(concentration.shape) == 1:
            concentration = np.expand_dims(concentration, axis=0)
            result = np.zeros((1, self.n_species))
        else:
            result = np.zeros((concentration.shape[0], self.n_species))
        delta = concentration[:, self.type_from1] * concentration[:, self.type_from2]
        result[:, self.type_from1] = -delta
        result[:, self.type_from2] = -delta
        result[:, self.type_to] = delta
        return result.squeeze()


class FissionReaction(object):
    def __init__(self, type_from, type_to1, type_to2, n_species):
        self.type_from = type_from
        self.type_to1 = type_to1
        self.type_to2 = type_to2
        self.n_species = n_species

    def __call__(self, concentration):
        if len(concentration.shape) == 1:
            concentration = np.expand_dims(concentration, axis=0)
            result = np.zeros((1, self.n_species))
        else:
            result = np.zeros((concentration.shape[0], self.n_species))
        delta = concentration[:, self.type_from]
        result[:, self.type_from] = -delta
        result[:, self.type_to1] = delta
        result[:, self.type_to2] = delta
        return result.squeeze()


class BasisFunctionConfiguration(object):
    def __init__(self, n_species):
        self._basis_functions = []
        self._n_species = n_species

    @property
    def functions(self):
        return self._basis_functions

    @property
    def n_basis_functions(self):
        return len(self._basis_functions)

    def add_conversion(self, type1, type2):
        self._basis_functions.append(ConversionReaction(type1, type2, self._n_species))

    def add_fusion(self, type_from1, type_from2, type_to):
        self._basis_functions.append(FusionReaction(type_from1, type_from2, type_to, self._n_species))

    def add_fission(self, type_from, type_to1, type_to2):
        self._basis_functions.append(FissionReaction(type_from, type_to1, type_to2, self._n_species))


class ReaDDyElasticNetEstimator(BaseEstimator):
    def __init__(self, trajs, basis_function_configuration, scale, alpha=1.0, l1_ratio=1.0, init_xi=None,
                 verbose=False, maxiter=15000, approx_jac=False, method='SLSQP'):
        """

        :param trajs:
        :param basis_function_configuration:
        :param scale:
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
        self.scale = scale
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

    def _get_slice(self, X):
        if X is not None:
            if isinstance(X, tuple) and len(X) == 2 and len(self.trajs) > 1:
                data = self.trajs[X[0]].counts[X[1]]
                expected = self.trajs[X[0]].dcounts_dt[X[1]]
            else:
                data = self.trajs[0].counts[X]
                expected = self.trajs[0].dcounts_dt[X]
        else:
            data = self.trajs[0].counts
            expected = self.trajs[0].dcounts_dt
        data = np.ascontiguousarray(data)
        expected = np.ascontiguousarray(expected)
        return data, expected

    def fit(self, X, y=None):
        """
        :param X: the counts
        :param y: counts time derivative
        :param kwargs:
        :return:
        """
        data, expected = self._get_slice(X)

        large_theta = np.array([f(data) for f in self.basis_function_configuration.functions])
        large_theta = np.ascontiguousarray(np.transpose(large_theta, axes=(1, 0, 2)))

        bounds = [(0., None)] * self.basis_function_configuration.n_basis_functions
        init_xi = self.init_xi

        jac = False if self.approx_jac else \
            lambda x: opt.elastic_net_objective_fun_jac(x, self.alpha, self.l1_ratio, large_theta, expected,
                                                        self.scale) / 1e6
        options = {'disp': False}
        if self.method == 'L-BFGS-B':
            options['maxiter'] = self.maxiter
            options['maxfun'] = self.maxiter
        result = so.minimize(
            lambda x: opt.elastic_net_objective_fun(x, self.alpha, self.l1_ratio, large_theta, expected,
                                                    self.scale) / 1e6,
            init_xi,
            bounds=bounds,
            tol=1e-16,
            method=self.method,
            jac=jac,
            options=options)

        self.coefficients_ = result.x

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


class CV(object):
    def __init__(self, traj, bfc, scale, alphas, l1_ratios, n_splits, init_xi, n_jobs=8, show_progress=True,
                 mode='k_fold', verbose=False, method='SLSQP'):
        self.alphas = alphas
        self.l1_ratios = l1_ratios
        self.n_jobs = n_jobs
        self.n_splits = n_splits
        self.traj = traj
        self.bfc = bfc
        self.scale = scale
        self.show_progress = show_progress
        self.result = []
        self.init_xi = init_xi
        self.mode = mode
        self.verbose = verbose
        self.method = method

    def compute_cv_result(self, params):
        if self.mode == 'k_fold':
            kf = KFold(n_splits=self.n_splits)
        elif self.mode == 'time_series_split':
            kf = TimeSeriesSplit(n_splits=self.n_splits)
        else:
            print("unknown mode: %s" % self.mode)
            return
        alpha, l1_ratio = params
        estimator = ReaDDyElasticNetEstimator(self.traj, self.bfc, self.scale, alpha=alpha,
                                              l1_ratio=l1_ratio, init_xi=self.init_xi, verbose=self.verbose,
                                              method=self.method)
        scores = []
        for train_idx, test_idx in kf.split(range(0, self.traj.n_time_steps)):
            estimator.fit(train_idx)
            if estimator.result_.success:
                scores.append(estimator.score(test_idx, self.traj.dcounts_dt[test_idx]))
        return {'scores': scores, 'alpha': alpha, 'l1_ratio': l1_ratio}

    def fit(self):
        params = itertools.product(self.alphas, self.l1_ratios)
        result = []
        if self.show_progress:
            from ipywidgets import IntProgress
            from IPython.display import display
            f = IntProgress(min=0, max=len(self.alphas) * len(self.l1_ratios) - 1)
            display(f)
        with Pool(processes=self.n_jobs) as p:
            for idx, res in enumerate(p.imap_unordered(self.compute_cv_result, params, 1)):
                result.append(res)
                if self.show_progress:
                    f.value = idx
        f.close()
        self.result = result


def get_dense_params(traj, bfc, scale, n_initial_values=16, n_jobs=8, initial_value=None):
    from ipywidgets import IntProgress
    from IPython.display import display
    if initial_value is not None:
        n_initial_values = 1
        initial_values = [initial_value]
    else:
        initial_values = [np.random.random(bfc.n_basis_functions) for _ in range(n_initial_values)]

    f = IntProgress(min=1, max=n_initial_values)
    display(f)

    def worker(init_xi):
        est = ReaDDyElasticNetEstimator(traj, bfc, scale, alpha=0, l1_ratio=1.0, init_xi=init_xi, verbose=False)
        est.fit(range(0, traj.n_time_steps))
        return est.coefficients_

    coeffs = []
    with Pool(processes=n_jobs) as p:
        for idx, coeff in enumerate(p.imap_unordered(worker, initial_values, 1)):
            coeffs.append(coeff)
            f.value = idx + 1
    f.close()
    return coeffs
