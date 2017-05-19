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
Created on 17.05.17

@author: clonker
"""

import analyze_tools as at
import functools
import operator

import h5py
import numpy as np
import os
import scipy.optimize as so
from pathos.multiprocessing import Pool


def get_count_trajectory(fname, cache_fname=None):
    if cache_fname and os.path.exists(cache_fname):
        return np.load(cache_fname)
    else:
        dtraj = []
        if os.path.exists(fname):
            with h5py.File(fname) as f:
                traj = f["readdy/trajectory"]
                traj_time = traj["time"]
                traj_time_records = traj["records"]
                for time, records in zip(traj_time, traj_time_records):
                    current_counts = [0] * 4
                    for record in records:
                        type_id = record["typeId"]
                        current_counts[type_id] += 1
                    dtraj.append(current_counts)
        else:
            print("file {} did not exist".format(fname))
        dtraj = np.array(dtraj)
        if cache_fname:
            np.save(cache_fname, dtraj)
        return dtraj


def frobenius_l1_regression(alpha, n_time_steps, n_basis_functions, n_species, theta, dcounts_dt, scale=None):
    bounds = [(0., None)] * n_basis_functions
    init_xi = np.array([.5] * n_basis_functions)
    iterations = []
    if not scale:
        scale = 1. / (n_time_steps * n_species)
    result = so.minimize(
        lambda x: at.lasso_minimizer_objective_fun(x, alpha, scale * theta, scale * dcounts_dt),
        init_xi,
        bounds=bounds,
        callback=lambda x: iterations.append(x),
        tol=1e-16,
        method='L-BFGS-B')
    return result.x


class Trajectory(object):
    def __init__(self, fname):
        with h5py.File(fname) as f:
            self._counts = f["readdy/observables/n_particles/data"][:].astype(np.double)
            self._box_size = [15., 15., 15.]
            self._time_step = .01
            self._thetas = []
            self._thetas_ode = []
            self._large_theta = None
            self._large_theta_norm_squared = 0
            self._n_basis_functions = 0
            self._n_time_steps = 0
            self._n_species = 0
            self._dcounts_dt = None
            self._xi = None
            self._dirty = True

    def rate_info(self, xi, diffusion_coefficient=.2, microscopic_rate=.05, reaction_radius=.7):
        self.update()
        tmp = np.sqrt(microscopic_rate / diffusion_coefficient) * reaction_radius
        rate_chapman = 4. * np.pi * diffusion_coefficient * reaction_radius * (1. - np.tanh(tmp) / tmp)
        rate_per_volume = xi * functools.reduce(operator.mul, self._box_size, 1)

        print("erban chapman rate (per volume): {}".format(rate_chapman))
        print("lasso fitted rate (per counts): {}".format(xi))
        print("lasso fitted rate (per volume): {}".format(rate_per_volume))

        return rate_chapman, xi, rate_per_volume

    def update(self):
        if self._dirty:
            self._dirty = False
            print("max counts = {}, min nonzero counts = {}".format(np.max(self.counts),
                                                                    np.min(self.counts[np.nonzero(self.counts)])))
            self._xi = None
            self._n_basis_functions = len(self._thetas)
            if len(self._thetas) > 0:
                self._n_time_steps = self.counts.shape[0]
                self._n_species = self.counts.shape[1]
                self._dcounts_dt = np.gradient(self.counts, axis=0) / self._time_step
                self._last_alpha = .01
                self._large_theta = np.array([f(self._counts) for f in self._thetas])
                self._large_theta = np.transpose(self._large_theta, axes=(1, 0, 2))
                self._large_theta_norm_squared = at.theta_norm_squared(self._large_theta)

    def __str__(self):
        self.update()
        string = "Trajectory("
        string += "counts.shape={}, box_size={}, time_step={}, n_basis_functions={}, large_theta.shape={}, " \
                  "n_time_steps={}, n_species={}, dirty={}, dcounts_dt.shape={}".format(self.counts.shape,
                                                                                        self._box_size, self.time_step,
                                                                                        self.n_basis_functions,
                                                                                        self._large_theta.shape,
                                                                                        self._n_time_steps,
                                                                                        self._n_species, self._dirty,
                                                                                        self._dcounts_dt.shape)
        string += ")"
        return string

    def frob_l1_regression(self, alpha, scale=None):
        self.update()
        return frobenius_l1_regression(alpha, self.n_time_steps, self.n_basis_functions, self.n_species,
                                       self._large_theta, self.dcounts_dt, scale)

    def estimate(self, alpha):
        self.update()
        self._xi = self.frob_l1_regression(alpha)
        return self._xi

    @property
    def propensities(self):
        if not self._xi:
            self.estimate(self._last_alpha)
        return self._xi

    @property
    def n_basis_functions(self):
        return self._n_basis_functions

    @property
    def thetas(self):
        return self._thetas

    @thetas.setter
    def thetas(self, value):
        self._thetas = value
        self._dirty = True

    @property
    def dcounts_dt(self):
        return self._dcounts_dt

    @property
    def counts(self):
        return self._counts

    @property
    def n_time_steps(self):
        return self._n_time_steps

    @counts.setter
    def counts(self, value):
        self._counts = value
        self._dirty = True

    @property
    def thetas(self):
        return self._thetas

    @property
    def time_step(self):
        return self._time_step

    @time_step.setter
    def time_step(self, value):
        self._time_step = value
        self._dirty = True

    def add_conversion(self, type1, type2):
        def conversion(concentration):
            if len(concentration.shape) == 1:
                concentration = np.expand_dims(concentration, axis=0)
                result = np.zeros((1, self._n_species))
            else:
                result = np.zeros((concentration.shape[0], self._n_species))
            result[:, type1] = -concentration[:, type1]
            result[:, type2] = concentration[:, type1]
            return result.squeeze()

        self._thetas.append(conversion)
        self._dirty = True

    def add_fusion(self, type_from1, type_from2, type_to):
        def fusion(concentration):
            if len(concentration.shape) == 1:
                concentration = np.expand_dims(concentration, axis=0)
                result = np.zeros((1, self._n_species))
            else:
                result = np.zeros((concentration.shape[0], self._n_species))
            delta = concentration[:, type_from1] * concentration[:, type_from2]
            result[:, type_from1] = -delta
            result[:, type_from2] = -delta
            result[:, type_to] = delta
            return result.squeeze()

        self._thetas.append(fusion)
        self._dirty = True

    def add_fission(self, type_from, type_to1, type_to2):
        def fission(concentration):
            if len(concentration.shape) == 1:
                concentration = np.expand_dims(concentration, axis=0)
                result = np.zeros((1, self._n_species))
            else:
                result = np.zeros((concentration.shape[0], self._n_species))
            delta = concentration[:, type_from]
            result[:, type_from] = -delta
            result[:, type_to1] = delta
            result[:, type_to2] = delta
            return result.squeeze()

        self._thetas.append(fission)
        self._dirty = True

    @property
    def n_species(self):
        return self._n_species


class CVResult(object):
    def __init__(self):
        self._large_theta_train = None
        self._train_data_derivative = None
        self._large_theta_test = None
        self._test_data_derivative = None
        self._costs_train = None
        self._costs_test = None
        self._relative_cost = None
        self._alphas = None
        self._coefficients = None

    @property
    def large_theta_train(self):
        return self._large_theta_train

    @large_theta_train.setter
    def large_theta_train(self, value):
        self._large_theta_train = value

    @property
    def train_data_derivative(self):
        return self._train_data_derivative

    @train_data_derivative.setter
    def train_data_derivative(self, value):
        self._train_data_derivative = value

    @property
    def large_theta_test(self):
        return self._large_theta_test

    @large_theta_test.setter
    def large_theta_test(self, value):
        self._large_theta_test = value

    @property
    def test_data_derivative(self):
        return self._test_data_derivative

    @test_data_derivative.setter
    def test_data_derivative(self, value):
        self._test_data_derivative = value

    @property
    def costs_train(self):
        return self._costs_train

    @costs_train.setter
    def costs_train(self, value):
        self._costs_train = value

    @property
    def costs_test(self):
        return self._costs_test

    @costs_test.setter
    def costs_test(self, value):
        self._costs_test = value

    @property
    def alphas(self):
        return self._alphas

    @alphas.setter
    def alphas(self, value):
        self._alphas = value

    @property
    def relative_cost(self):
        return self._relative_cost

    @relative_cost.setter
    def relative_cost(self, value):
        self._relative_cost = value

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        self._coefficients = value


def magnitude(x):
    return np.math.floor(np.math.log10(x))


class CV(object):
    def __init__(self, traj_train, traj_test=None):
        if not traj_test:
            traj_test = traj_train
        traj_train.update()
        traj_test.update()
        assert traj_train.time_step == traj_test.time_step
        assert traj_train.n_basis_functions == traj_test.n_basis_functions
        self._traj_train = traj_train
        self._traj_test = traj_test
        self._scale = 1. / (traj_train.n_species * traj_train.n_time_steps)

    def find_alpha(self, n_grid_points=200, train_indices=range(0, 6000), test_indices=range(6000, 12000),
                   return_cv_result=False, njobs=8, alphas=None):
        if not alphas:
	    result = self.calculate_cost([0], train_indices, test_indices)
        	norm_of_coeff = np.linalg.norm(result.coefficients[0], ord=1)
        	print("norm of coefficients for alpha=0: {}".format(norm_of_coeff))
        	quotient = result.costs_test[0] / norm_of_coeff
        	print("quotient = {}, order of magnitude = {}".format(quotient, magnitude(quotient)))

        	alphas = np.linspace(0, 10 ** (magnitude(quotient) + 1), num=n_grid_points)
        cv_result = self.calculate_cost(alphas, train_indices, test_indices, njobs)
        min_idx = np.argmin(cv_result.costs_test)
        print("best suited alpha found at idx={}, alpha={}, costs_test={}".format(min_idx, cv_result.alphas[min_idx],
                                                                                  cv_result.costs_test[min_idx]))
        if return_cv_result:
            return cv_result.alphas[min_idx], cv_result
        else:
            return cv_result.alphas[min_idx]

    def calculate_cost(self, alphas, train_indices, test_indices, njobs=8):
        self._traj_train.update()
        cv = CVResult()
        cv.alphas = alphas

        cv.large_theta_train = np.array([f(self._traj_train.counts[train_indices]) for f in self._traj_train.thetas])
        cv.large_theta_train = np.transpose(cv.large_theta_train, axes=(1, 0, 2))
        cv.train_data_derivative = self._traj_train.dcounts_dt[train_indices]

        cv.large_theta_test = np.array([f(self._traj_test.counts[test_indices]) for f in self._traj_test.thetas])
        cv.large_theta_test = np.transpose(cv.large_theta_test, axes=(1, 0, 2))
        cv.test_data_derivative = self._traj_test.dcounts_dt[test_indices]

        with Pool(processes=njobs) as p:
            coefficients = p.map(
                lambda x: frobenius_l1_regression(x, self._traj_train.n_time_steps, self._traj_train.n_basis_functions,
                                                  self._traj_train.n_species, cv.large_theta_train,
                                                  cv.train_data_derivative, scale=self._scale),
                alphas)
        cv.coefficients = coefficients
        cost_learn = []
        cost_test = []
        relative_cost = []
        for coeff in coefficients:
            cost_learn.append(at.lasso_minimizer_objective_fun(coeff, 0.0, cv.large_theta_train * self._scale,
                                                               cv.train_data_derivative * self._scale))
            cost_test.append(at.lasso_minimizer_objective_fun(coeff, 0.0, cv.large_theta_test * self._scale,
                                                              cv.test_data_derivative * self._scale))
            relative_cost.append(cost_test[-1] / at.theta_norm_squared(cv.large_theta_test))
        cv.costs_test = cost_test
        cv.costs_train = cost_learn
        cv.relative_cost = relative_cost
        return cv


def preprocess_data(X, y, normalize):
    """
    X = (X - X_offset) / X_scale
    """

    X_offset = np.average(X, axis=0)
    X -= X_offset
    if normalize:
        from sklearn.preprocessing.data import normalize
        X, X_scale = normalize(X, axis=0, copy=False, return_norm=True)
    else:
        X_scale = np.ones(X.shape[1])
    y_offset = np.average(y, axis=0)
    y = y - y_offset

    return X, y, X_offset, y_offset, X_scale
