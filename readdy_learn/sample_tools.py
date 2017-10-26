import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from pathos.multiprocessing import Pool

import readdy_learn.analyze.tools as pat
from readdy_learn.analyze.sklearn import ReaDDyElasticNetEstimator


class Suite(object):
    def __init__(self, set_up_system, bfc, alpha=0., l1_ratio=1., maxiter=30000, tol=1e-12, interp_degree=10,
                 init_xi=None, system=None, trajectory=None):
        self._set_up_system = set_up_system
        self._trajectory = trajectory
        self._bfc = bfc
        self._alpha = alpha
        self._l1_ratio = l1_ratio
        self._maxiter = maxiter
        self._tol = tol
        self._interp_degree = interp_degree
        self._init_xi = init_xi
        self._system = system

    @classmethod
    def from_trajectory(cls, trajectory, system, bfc, alpha=0., l1_ratio=1., maxiter=30000, tol=1e-12,
                        interp_degree=10, init_xi=None):
        return Suite(None, bfc, alpha=alpha, l1_ratio=l1_ratio, maxiter=maxiter, tol=tol, interp_degree=interp_degree,
                     init_xi=init_xi, system=system, trajectory=trajectory)

    @classmethod
    def from_generator(cls, system_generator, bfc, alpha=0., l1_ratio=1., maxiter=30000, tol=1e-12,
                       interp_degree=10, init_xi=None):
        return Suite(system_generator, bfc=bfc, alpha=alpha, l1_ratio=l1_ratio, maxiter=maxiter, tol=tol,
                     interp_degree=interp_degree, init_xi=init_xi)

    def get_estimator(self, sys, bfc, timestep, interp_degree=10, verbose=False):
        counts, times, config = sys.get_counts_config(timestep=timestep)

        traj = pat.Trajectory.from_counts(counts, times[1] - times[0], interp_degree=interp_degree,
                                          verbose=verbose)
        traj.update()

        est = ReaDDyElasticNetEstimator(traj, bfc, alpha=self._alpha, l1_ratio=self._l1_ratio,
                                        maxiter=self._maxiter, method='SLSQP', verbose=verbose, approx_jac=False,
                                        options={'ftol': self._tol}, rescale=False, init_xi=self._init_xi)
        return est

    def run(self, sys, bfc, verbose=True, n_frames=None, timestep=None):
        if verbose:
            print("---- begin suite run")
        counts, times, config = sys.get_counts_config(n_frames=n_frames, timestep=timestep)

        traj = pat.Trajectory.from_counts(counts, times[1] - times[0], verbose=verbose,
                                          interp_degree=self._interp_degree)
        traj.update()

        est = ReaDDyElasticNetEstimator(traj, bfc, alpha=self._alpha, l1_ratio=self._l1_ratio,
                                        maxiter=self._maxiter, method='SLSQP', verbose=verbose, approx_jac=False,
                                        options={'ftol': self._tol}, rescale=False, init_xi=self._init_xi)
        est.fit(None)
        if verbose:
            print("---- finish suite run, success = {}".format(est.success_))
        if est.success_:
            coefficients = est.coefficients_
            return timestep, coefficients
        else:
            return timestep, None

    @staticmethod
    def estimated_behavior(coefficients, bfc, initial_counts, times):
        def fun(data, _):
            theta = np.array([f(data) for f in bfc.functions])
            return np.matmul(coefficients, theta)

        estimated_realisation = odeint(fun, initial_counts, times)
        return estimated_realisation

    def plot_concentrations(self, system, timestep):
        counts, times, config = system.get_counts_config(timestep=timestep)

        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        for t in config.types.keys():
            type_id = config.types[t]
            # ax1.plot(times, estimated[:, type_id], "k--")
            ax1.plot(times, counts[:, type_id], label="counts " + t)
        ax1.legend(loc="best")

    def _get_system(self):
        if self._set_up_system is not None:
            return self._set_up_system()
        else:
            return self._system

    def plot(self, file):
        system = self._get_system()
        config = system.get_trajectory_config()

        f = np.load(file)
        data = f['rates'].item()
        counts = f['counts']
        times = f['times']
        xs = np.asarray([k for k in data.keys()])
        smallest_time_step = min(data.keys())

        estimated = self.estimated_behavior(np.mean(data[smallest_time_step], axis=0), self._bfc, counts[0], times)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        ax1.set_title('Concentration')
        for t in config.types.keys():
            type_id = config.types[t]
            ax1.plot(times, estimated[:, type_id], "k--")
            ax1.plot(times, counts[:, type_id], label="counts " + t)
        ax1.legend(loc="best")

        ax2.set_title('Estimated rates')
        for i, reaction in enumerate(system.reactions):
            ys, yerr = [], []
            for time_step in data.keys():
                rates = data[time_step]
                ys.append(np.mean(rates[:, i]))
                yerr.append(np.std(rates[:, i]))
            ax2.errorbar(xs, ys, yerr=yerr, label='estimated ' + str(reaction))
            ax2.plot(xs, reaction.rate * np.ones_like(xs), "--", label="expected " + str(reaction))

        ax2.set_xscale('log')
        ax2.set_xlabel("$\Delta t$")
        ax2.set_ylabel("rate")
        ax2.legend(loc="best")

        fig.show()
        plt.show()

    def calculate(self, file, timesteps, n_steps, n_realizations=1, write_concentrations_for_time_step=None,
                  verbose=True, save=True, njobs=8):
        if os.path.exists(file):
            raise ValueError("File already existed: {}".format(file))

        allrates = {}

        for k in timesteps:
            allrates[k] = []

        if write_concentrations_for_time_step is None:
            write_concentrations_for_time_step = min(timesteps)
        concentrations = None

        if njobs > 1:
            if verbose:
                print("---- suite calculate parallel")

            def run_wrapper(args):
                return self.run(**args)

            params = []
            for n in range(n_realizations):
                system = self._get_system()
                system.simulate(n_steps)
                for dt in timesteps:
                    params.append({'sys': system, 'bfc': self._bfc, 'timestep': dt, 'verbose': verbose})
                    if dt == write_concentrations_for_time_step:
                        counts, times, config = system.get_counts_config(timestep=dt)
                        concentrations = counts.squeeze(), times.squeeze()
            with Pool(processes=njobs) as p:
                for dt, rates in p.imap_unordered(run_wrapper, params, chunksize=1):
                    if rates is not None:
                        allrates[dt].append(rates)
        else:
            if verbose:
                print("---- suite calculate serial")
            for n in range(n_realizations):
                system = self._get_system()
                system.simulate(n_steps)
                if verbose:
                    print("suite calculate realization {} of {}, timesteps={}".format(n, n_realizations, timesteps))
                for dt in timesteps:
                    _, rates = self.run(system, self._bfc, timestep=dt, verbose=verbose)
                    if rates is not None:
                        allrates[dt].append(rates)
                        if dt == write_concentrations_for_time_step:
                            counts, times, config = system.get_counts_config(timestep=dt)
                            concentrations = counts.squeeze(), times.squeeze()
        for k in allrates.keys():
            allrates[k] = np.asarray(allrates[k])
        if save:
            print("writing {} rates with {} counts and {} times to {}".format(allrates, concentrations[0],
                                                                              concentrations[1], file))
            np.savez(file, rates=allrates, counts=concentrations[0], times=concentrations[1])
        return np.mean(allrates[min(timesteps)], axis=0)
