import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from pathos.multiprocessing import Pool

import readdy_learn.analyze.tools as pat
from readdy_learn.analyze.sklearn import ReaDDyElasticNetEstimator


class Suite(object):
    def __init__(self, set_up_system, alpha=0., l1_ratio=1., maxiter=30000, tol=1e-12, interp_degree=10):
        self._set_up_system = set_up_system
        self._alpha = alpha
        self._l1_ratio = l1_ratio
        self._maxiter = maxiter
        self._tol = tol
        self._interp_degree = interp_degree


    def get_estimator(self, sys, bfc, timestep, interp_degree=10, verbose=False):
        counts, times, config = sys.get_counts_config(timestep=timestep)

        traj = pat.Trajectory.from_counts(config, counts, times[1] - times[0], interp_degree=interp_degree,
                                          verbose=verbose)
        traj.update()

        est = ReaDDyElasticNetEstimator(traj, bfc, alpha=self._alpha, l1_ratio=self._l1_ratio,
                                        maxiter=self._maxiter, method='SLSQP', verbose=verbose, approx_jac=False,
                                        options={'ftol': self._tol})
        return est


    def run(self, sys, bfc, verbose=True, n_frames=None, timestep=None):
        counts, times, config = sys.get_counts_config(n_frames=n_frames, timestep=timestep)

        traj = pat.Trajectory.from_counts(config, counts, times[1] - times[0], verbose=verbose,
                                          interp_degree=self._interp_degree)
        traj.update()

        est = ReaDDyElasticNetEstimator(traj, bfc, alpha=self._alpha, l1_ratio=self._l1_ratio,
                                        maxiter=self._maxiter, method='SLSQP', verbose=verbose, approx_jac=False,
                                        options={'ftol': self._tol})
        est.fit(None)
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

    def plot(self, file):
        system, bfc = self._set_up_system()
        config = system.get_trajectory_config()

        f = np.load(file)
        data = f['rates'].item()
        counts = f['counts']
        times = f['times']
        xs = np.asarray([k for k in data.keys()])
        smallest_time_step = min(data.keys())

        estimated = self.estimated_behavior(np.mean(data[smallest_time_step], axis=0), bfc, counts[0], times)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

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
        ax2.set_xlabel("time step")
        ax2.set_ylabel("rate")
        ax2.legend(loc="best")

        fig.show()
        plt.show()

    def calculate(self, file, timesteps, n_steps, n_realizations=20, write_concentrations_for_time_step=None,
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
            def run_wrapper(args):
                return self.run(**args)

            params = []
            for n in range(n_realizations):
                system, bfc = self._set_up_system()
                system.simulate(n_steps)
                for dt in timesteps:
                    params.append({'sys': system, 'bfc': bfc, 'timestep': dt, 'verbose': verbose})
                    if dt == write_concentrations_for_time_step:
                        counts, times, config = system.get_counts_config(timestep=dt)
                        concentrations = counts.squeeze(), times.squeeze()
            with Pool(processes=njobs) as p:
                for dt, rates in p.imap_unordered(run_wrapper, params, chunksize=1):
                    if rates is not None:
                        allrates[dt].append(rates)
        else:
            for n in range(n_realizations):
                system, bfc = self._set_up_system()
                system.simulate(n_steps)
                for dt in timesteps:
                    _, rates = self.run(system, bfc, timestep=dt, verbose=verbose)
                    if rates is not None:
                        allrates[dt].append(rates)
                        if dt == write_concentrations_for_time_step:
                            counts, times, config = system.get_counts_config(timestep=dt)
                            concentrations = counts.squeeze(), times.squeeze()
        for k in allrates.keys():
            allrates[k] = np.asarray(allrates[k])
        if save:
            np.savez(file, rates=allrates, counts=concentrations[0], times=concentrations[1])
        for dt in timesteps:
            rates = allrates[dt]
            if verbose:
                print("got {:.3f}±{:.3f}  and {:.3f}±{:.3f} for timestep={}".format(
                    np.mean(rates[:, 0]), np.std(rates[:, 0]),
                    np.mean(rates[:, 1]), np.std(rates[:, 1]), dt))
        return np.mean(allrates[min(timesteps)], axis=0)
