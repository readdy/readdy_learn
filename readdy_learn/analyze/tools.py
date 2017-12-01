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

import functools
import operator
import os

import h5py
import numpy as np
# import readdy.util.io_utils as ioutils
import scipy.optimize as so
from pathos.multiprocessing import Pool

import readdy_learn.analyze.derivative as deriv
from readdy_learn import analyze_tools as opt

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


def frobenius_l1_regression(alpha, n_basis_functions, theta, dcounts_dt, scale=-1.):
    bounds = [(0., None)] * n_basis_functions
    init_xi = np.array([.5] * n_basis_functions)
    iterations = []
    fun = lambda x: opt.lasso_minimizer_objective_fun(x, alpha, theta, dcounts_dt, scale)
    result = so.minimize(
        fun,
        init_xi,
        bounds=bounds,
        callback=lambda x: iterations.append(x),
        tol=1e-16,
        method='L-BFGS-B')
    return result.x


class TrajectoryConfig(object):
    def __init__(self, file_path):
        self.types = ioutils.get_particle_types(file_path)
        self.reactions = ioutils.get_reactions(file_path)


class Trajectory(object):
    def __init__(self, counts, time_step, interpolation_degree=10, verbose=True,
                 ld_derivative_atol=1e-7, ld_derivative_rtol=1e-10, ld_derivative_alpha=3e-3,
                 ld_derivative_solver='lgmres', ld_derivative_maxit=10000, ld_derivative_linalg_solver_maxit=100,
                 ld_derivative_linalg_solver_tol=1e-4, ld_derivative_use_preconditioner=True,
                 fname=None):
        if isinstance(counts, str):
            fname = counts
            counts = None
        self._counts = counts
        self._box_size = [15., 15., 15.]
        self._time_step = time_step
        self._thetas = []
        self._thetas_ode = []
        self._large_theta = None
        self._large_theta_norm_squared = 0
        self._n_basis_functions = 0
        self._xi = None
        self._dirty = True
        self._verbose = verbose
        self._interpolation_degree = interpolation_degree
        self._fname = fname
        self._separate_derivs = {}
        self.ld_derivative_config = {'atol': ld_derivative_atol, 'rtol': ld_derivative_rtol,
                                     'alpha': ld_derivative_alpha, 'solver': ld_derivative_solver,
                                     'maxit': ld_derivative_maxit, 'linalg_solver_maxit': ld_derivative_linalg_solver_maxit,
                                     'tol': ld_derivative_linalg_solver_tol, 'precondition': ld_derivative_use_preconditioner}
        if fname is not None and os.path.exists(fname):
            archive = np.load(fname)
            self._counts = archive['counts']
            for s in range(self.n_species):
                if 'dcounts_dt_{}'.format(s) in archive.keys():
                    self._separate_derivs[s] = archive['dcounts_dt_{}'.format(s)]
            if 'dt' in archive.keys():
                self._time_step = archive['dt']

    @classmethod
    def from_file_name(cls, fname, time_step, interp_degree=10, verbose=True):
        with h5py.File(fname) as f:
            return Trajectory(f["readdy/observables/n_particles/data"][:].astype(np.double), time_step,
                              verbose=verbose, interpolation_degree=interp_degree)

    @classmethod
    def from_counts(cls, counts, time_step, interp_degree=10, verbose=True):
        return Trajectory(counts, time_step, interpolation_degree=interp_degree, verbose=verbose)

    @property
    def interpolation_degree(self):
        return self._interpolation_degree

    @interpolation_degree.setter
    def interpolation_degree(self, value):
        self._interpolation_degree = value

    def rate_info(self, xi, diffusion_coefficient=.2, microscopic_rate=.05, reaction_radius=.7):
        self.update()
        tmp = np.sqrt(microscopic_rate / diffusion_coefficient) * reaction_radius
        rate_chapman = 4. * np.pi * diffusion_coefficient * reaction_radius * (1. - np.tanh(tmp) / tmp)
        rate_per_volume = xi * functools.reduce(operator.mul, self._box_size, 1)

        if self._verbose:
            print("erban chapman rate (per volume): {}".format(rate_chapman))
            print("lasso fitted rate (per counts): {}".format(xi))
            print("lasso fitted rate (per volume): {}".format(rate_per_volume))

        return rate_chapman, xi, rate_per_volume

    @property
    def separate_derivs(self):
        return self._separate_derivs

    def persist(self, **kw):
        if self._fname is not None:
            derivs = {}
            for s in range(self.n_species):
                if s in self._separate_derivs.keys():
                    derivs['dcounts_dt_{}'.format(s)] = self._separate_derivs[s]
            np.savez(self._fname, counts=self.counts, dt=self.time_step, **kw,
                     **derivs)
        else:
            raise ValueError("no file name set!")

    @property
    def fname(self):
        return self._fname

    @fname.setter
    def fname(self, value):
        self._fname = value

    def calculate_dX(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression as interp
        from scipy import optimize

        if self._interpolation_degree is None:
            return None

        if self.dcounts_dt is not None:
            return self.dcounts_dt

        is_gradient = False

        X = self.times
        interpolated = np.empty_like(self.counts)
        for s in range(self.n_species):
            counts = self.counts[:, s]
            indices = 1 + np.where(counts[:-1] != counts[1:])[0]
            indices = np.insert(indices, 0, 0)
            if indices[-1] != len(counts) -1:
                indices = np.append(indices, len(counts) - 1)
            if s in self.separate_derivs:
                interpolated[:, s] = self.separate_derivs[s]
                is_gradient = True
            elif self._interpolation_degree == 'pw_linear':
                interpolated[:, s] = np.interp(X, X[indices], counts[indices])
                is_gradient = False
            elif self._interpolation_degree == 'FD':
                wwidth = 2
                iix = 0
                for ix, widx in enumerate(deriv.window_evensteven(indices, width=wwidth)):
                    iix_curr = indices[ix]
                    x = X[iix_curr]
                    coeff = deriv.fd_coeff(x, X[widx], k=1)
                    d = coeff.dot(counts[widx])
                    interpolated[iix:iix_curr, s] = d
                    iix = iix_curr
                is_gradient = True
            elif self._interpolation_degree == 'regularized_derivative':
                print('---- calculating regularized derivative for species {} with {} data points ----'
                      .format(s, len(indices)))
                dX = deriv.ld_derivative(data=counts[indices], xs=X[indices], verbose=False,
                                         show_progress=True, **self.ld_derivative_config)
                prev = 0
                for ix in range(len(indices)):
                    interpolated[indices[prev]:(indices[ix]+1), s] = dX[ix]
                    prev = ix
                is_gradient = True
            elif self._interpolation_degree < 0:
                fun = lambda t, b, c, d, e, g: b * np.exp(c * t) + d * t + e * t * t + g * t * t * t
                dfun_db = lambda t, b, c, d, e, g: np.exp(c * t)
                dfun_dc = lambda t, b, c, d, e, g: t * b * np.exp(c * t)
                dfun_dd = lambda t, b, c, d, e, g: t
                dfun_de = lambda t, b, c, d, e, g: t * t
                dfun_dg = lambda t, b, c, d, e, g: t * t * t
                derivatives = [dfun_db, dfun_dc, dfun_dd, dfun_de, dfun_dg]

                def jac(t, b, c, d, e, g):
                    result = np.array([np.array(f(t, b, c, d, e, g)) for f in derivatives])
                    return result.T

                copt, _ = optimize.curve_fit(fun, X, counts, maxfev=300000, jac=jac)

                ff = lambda t: fun(t, *copt)
                dff = lambda t: jac(t, *copt)
                interpolated[:, s] = ff(X)
                is_gradient = False
            else:
                poly_feat = PolynomialFeatures(degree=self._interpolation_degree)
                regression = interp()
                pipeline = Pipeline([("poly", poly_feat), ("regression", regression)])
                pipeline.fit(X[indices, np.newaxis], counts[indices])

                ys = pipeline.predict(X[:, np.newaxis])
                interpolated[:, s] = ys
                is_gradient = False
            if not is_gradient:
                interpolated[:, s] = np.gradient(interpolated, axis=0) / self._time_step
        return interpolated

    def update(self):
        if self._dirty:
            self._dirty = False
            if self._verbose:
                print("max counts = {}, min nonzero counts = {}".format(np.max(self.counts),
                                                                        np.min(self.counts[np.nonzero(self.counts)])))
            self._xi = None
            self._n_basis_functions = len(self._thetas)

            # todo this is garbage: np.gradient(self.counts, axis=0) / self._time_step
            if len(self._thetas) > 0:
                self._last_alpha = .01
                self._large_theta = np.array([f(self._counts) for f in self._thetas])
                self._large_theta = np.transpose(self._large_theta, axes=(1, 0, 2))
                self._large_theta_norm_squared = opt.theta_norm_squared(self._large_theta)

    def __str__(self):
        self.update()
        string = "Trajectory("
        string += "counts.shape={}, box_size={}, time_step={}, n_basis_functions={}, large_theta.shape={}, " \
                  "n_time_steps={}, n_species={}, dirty={}" \
            .format(self.counts.shape, self._box_size, self.time_step, self.n_basis_functions, self._large_theta.shape,
                    self.n_time_steps, self.n_species, self._dirty)
        string += ")"
        return string

    def frob_l1_regression(self, alpha, scale=-1.):
        self.update()
        return frobenius_l1_regression(alpha, self.n_basis_functions, self._large_theta, self.dcounts_dt, scale=scale)

    def estimate(self, alpha):
        self.update()
        self._xi = self.frob_l1_regression(alpha)
        return self._xi

    @property
    def times(self):
        times = np.linspace(0, self.counts.shape[0] * self.time_step, endpoint=False, num=self.counts.shape[0])
        assert times.shape[0] == self.counts.shape[0]
        assert np.isclose(times[1] - times[0], self.time_step), "times[1]-times[0] was {} but was expected to be {}" \
            .format(times[1] - times[0], self.time_step)
        return times

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
        if len(self.separate_derivs.keys()) != self.n_species:
            print("Dont have derivative (got {} but need {})".format(len(self.separate_derivs.keys()), self.n_species))
            return None
        deriv = np.empty_like(self.counts)
        for s in range(self.n_species):
            deriv[:, s] = self.separate_derivs[s]
        return deriv

    @dcounts_dt.setter
    def dcounts_dt(self, value):
        for s in range(self.n_species):
            self.separate_derivs[s] = value[:, s]

    @property
    def theta(self):
        return self._large_theta

    @property
    def counts(self):
        return self._counts

    @property
    def n_time_steps(self):
        return self.counts.shape[0]

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
                result = np.zeros((1, self.n_species))
            else:
                result = np.zeros((concentration.shape[0], self.n_species))
            result[:, type1] = -concentration[:, type1]
            result[:, type2] = concentration[:, type1]
            return result.squeeze()

        self._thetas.append(conversion)
        self._dirty = True

    def add_fusion(self, type_from1, type_from2, type_to):
        def fusion(concentration):
            if len(concentration.shape) == 1:
                concentration = np.expand_dims(concentration, axis=0)
                result = np.zeros((1, self.n_species))
            else:
                result = np.zeros((concentration.shape[0], self.n_species))
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
                result = np.zeros((1, self.n_species))
            else:
                result = np.zeros((concentration.shape[0], self.n_species))
            delta = concentration[:, type_from]
            result[:, type_from] = -delta
            result[:, type_to1] = delta
            result[:, type_to2] = delta
            return result.squeeze()

        self._thetas.append(fission)
        self._dirty = True

    @property
    def n_species(self):
        return self.counts.shape[1]


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
        self._scale = 1. / (2. * traj_train.n_species * traj_train.n_time_steps)

    def _find_alpha_recurse(self, level, search_interval, train_indices, test_indices, return_cv_result, njobs,
                            n_grid_points, max_level):
        alphas = np.linspace(search_interval[0], search_interval[1], num=n_grid_points)
        spacing = alphas[1] - alphas[0]

        cv_result = self.calculate_cost(alphas, train_indices, test_indices, njobs)
        min_idx = np.argmin(cv_result.costs_test)
        print("best suited alpha found on level {} at idx={}, alpha={}, costs_test={}"
              .format(level, min_idx, cv_result.alphas[min_idx], cv_result.costs_test[min_idx]))

        if level >= max_level:
            if return_cv_result:
                return cv_result.alphas[min_idx], cv_result
            else:
                return cv_result.alphas[min_idx]
        else:
            # take best alpha grid point and make interval around that
            new_interval = [max(0, alphas[min_idx] - .5 * spacing), alphas[min_idx] + .5 * spacing]
            return self._find_alpha_recurse(level + 1, new_interval, train_indices, test_indices, return_cv_result,
                                            njobs, n_grid_points, max_level)

    def find_alpha_recurse(self, n_grid_points=10, train_indices=range(0, 6000), test_indices=range(6000, 12000),
                           return_cv_result=False, njobs=8, max_level=4, initial_interval=None):
        assert n_grid_points > 1, "number of grid points should be larger than 1"
        if initial_interval is None:
            result = self.calculate_cost([0], train_indices, test_indices)
            norm_of_coeff = np.linalg.norm(result.coefficients[0], ord=1)
            print("norm of coefficients for alpha=0: {}".format(norm_of_coeff))
            quotient = self._scale * (result.costs_test[0] * result.costs_test[0]) / norm_of_coeff
            print("quotient = {}, order of magnitude = {}".format(quotient, magnitude(quotient)))
            initial_interval = [0, 10 ** (magnitude(quotient) + 1)]

        return self._find_alpha_recurse(0, initial_interval, train_indices, test_indices, return_cv_result,
                                        njobs, n_grid_points, max_level)

    def find_alpha(self, n_grid_points=200, train_indices=range(0, 6000), test_indices=range(6000, 12000),
                   return_cv_result=False, njobs=8, alphas=None):
        if alphas is None:
            result = self.calculate_cost([0], train_indices, test_indices)
            norm_of_coeff = np.linalg.norm(result.coefficients[0], ord=1)
            print("norm of coefficients for alpha=0: {}".format(norm_of_coeff))
            quotient = self._scale * (result.costs_test[0] * result.costs_test[0]) / norm_of_coeff
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
                lambda x: frobenius_l1_regression(x, self._traj_train.n_basis_functions, cv.large_theta_train,
                                                  cv.train_data_derivative, scale=self._scale),
                alphas)
        cv.coefficients = coefficients
        cost_learn = []
        cost_test = []
        relative_cost = []
        for coeff in coefficients:
            cost_learn.append(opt.score(coeff, cv.large_theta_train, cv.train_data_derivative))
            cost_test.append(opt.score(coeff, cv.large_theta_test, cv.test_data_derivative))
            relative_cost.append(cost_test[-1] / np.math.sqrt(opt.theta_norm_squared(cv.large_theta_test)))
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
