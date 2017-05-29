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

import numbers

import analyze_tools.opt as opt
import numpy as np
import scipy.optimize as so
from sklearn.linear_model.base import BaseEstimator


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
    def __init__(self, trajs, basis_function_configuration, scale, alpha=1.0, l1_ratio=1.0, init_xi = None):
        self.basis_function_configuration = basis_function_configuration
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.scale = scale
        if not isinstance(trajs, (list, tuple)):
            self.trajs = [trajs]
        else:
            self.trajs = trajs
        if init_xi is None:
            self.init_xi = np.array([.5]*self.basis_function_configuration.n_basis_functions)
        else:
            self.init_xi = init_xi

    def _get_slice(self, X):
        if X is not None:
            if isinstance(X, tuple) and len(X)==2 and len(self.trajs) > 1:
                data = self.trajs[X[0]].counts[X[1]]
                expected = self.trajs[X[0]].dcounts_dt[X[1]]
            else:
                data = self.trajs[0].counts[X]
                expected = self.trajs[0].dcounts_dt[X]
        else:
            data = self.trajs[0].counts
            expected = self.trajs[0].dcounts_dt
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
        large_theta = np.transpose(large_theta, axes=(1, 0, 2))

        bounds = [(0., None)] * self.basis_function_configuration.n_basis_functions
        init_xi = self.init_xi
        iterations = []
        fun = lambda x: opt.elastic_net_objective_fun(x, self.alpha, self.l1_ratio, large_theta, expected, self.scale)

        result = so.minimize(
            fun,
            init_xi,
            bounds=bounds,
            callback=lambda x: iterations.append(x),
            jac=False,
            tol=1e-16,
            method='L-BFGS-B')

        self.coefficients_ = result.x

        return self

    def score(self, X, y):
        data, _ = self._get_slice(X)
        large_theta = np.array([f(data) for f in self.basis_function_configuration.functions])
        large_theta = np.transpose(large_theta, axes=(1, 0, 2))
        return -1.*opt.score(self.coefficients_, large_theta, y)
