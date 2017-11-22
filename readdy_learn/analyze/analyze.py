import os

import matplotlib.pyplot as plt
import numpy as np

import readdy_learn.analyze.derivative as deriv
import readdy_learn.analyze.estimator as rlas
import readdy_learn.analyze.generate as generate
import readdy_learn.analyze.tools as tools
import readdy_learn.sample_tools as sample_tools


def estimate_noise_variance(xs, ys):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression as interp

    poly_feat = PolynomialFeatures(degree=6)
    regression = interp()  # interp(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], n_alphas=100)
    pipeline = Pipeline([("poly", poly_feat), ("regression", regression)])
    pipeline.fit(xs[:, np.newaxis], ys)

    ff = lambda t: pipeline.predict(t[:, np.newaxis])
    return np.var(ff(xs) - ys, ddof=0), ff


def obtain_derivative(traj, desired_n_counts=6000, alpha=1000, atol=1e-10, tol=1e-10, maxit=1000, alpha_search_depth=5,
                      interp_degree='regularized_derivative', variance=None, verbose=False, njobs=8):
    if traj.dcounts_dt is None:
        if interp_degree == 'regularized_derivative':
            interp_degree = traj.interpolation_degree
            traj.interpolation_degree = None
            traj.update()
            stride = max(traj.counts.shape[0] // desired_n_counts, 1)
            strided_times = traj.times[::stride]
            strided_counts = traj.counts[::stride, :]
            strided_dt = strided_times[1] - strided_times[0]
            print("got {} counts (and {} corresp. time steps), dt=".format(strided_counts.shape[0], len(strided_times)),
                  strided_dt)

            dx = np.empty(shape=(len(traj.times), traj.n_species))

            used_alphas = []
            for species in range(traj.n_species):
                ys = strided_counts[:, species]
                kw = {'maxit': maxit, 'linalg_solver_maxit': 1000000, 'tol': tol, 'atol': atol, 'rtol': None,
                      'precondition': False, 'solver': 'bicgstab', 'verbose': verbose}
                if isinstance(alpha, np.ndarray):
                    if len(alpha) > 1:
                        best_alpha, ld = deriv.best_ld_derivative(ys, strided_times, alpha, n_iters=alpha_search_depth,
                                                                  variance=variance, njobs=njobs, **kw)
                    else:
                        alpha = alpha[0]
                        ld = deriv.ld_derivative(ys, strided_times, alpha=alpha, **kw)
                        best_alpha = alpha
                else:
                    ld = deriv.ld_derivative(ys, strided_times, alpha=alpha, **kw)
                    best_alpha = alpha
                # linearly interpolate to the full time range
                integrated_ld = deriv.integrate.cumtrapz(ld, x=strided_times, initial=0) + ys[0]
                if variance is not None:
                    var = variance
                else:
                    var, ff = estimate_noise_variance(strided_times, ys)
                print("MSE =", deriv.mse(integrated_ld, ys), "noise variance =", var)
                dx[:, species] = np.interp(traj.times, strided_times, ld)

                used_alphas.append(best_alpha)
            traj.dcounts_dt = dx
            traj.interpolation_degree = interp_degree
            traj.persist(alpha=used_alphas)
            return used_alphas, traj
        else:
            traj.update()
            traj.persist()
            return [], traj
    else:
        print("traj already contains derivatives, skip this")
        return [], traj


class ReactionAnalysis(object):
    def __init__(self, bfc, desired_rates, initial_states, set_up_system, recompute=False, recompute_traj=False,
                 fname_prefix="", fname_postfix="", n_species=4, target_n_counts=150, timestep=5e-4,
                 interp_degree='regularized_derivative', ld_derivative_config=None):
        if initial_states is None or len(initial_states) <= 1:
            raise ValueError("needs at least two initial states!")
        if ld_derivative_config is None:
            ld_derivative_config = {
                'ld_derivative_atol': 1e-8,
                'ld_derivative_rtol': None,
                'ld_derivative_linalg_solver_maxit': 100000,
                'ld_derivative_alpha': 1e-1,
                'ld_derivative_solver': 'lgmres',
                'ld_derivative_linalg_solver_tol': 1e-10,
                'ld_derivative_use_preconditioner': False
            }
        self._ld_derivative_config = ld_derivative_config
        for state in initial_states:
            assert len(np.squeeze(state)) == n_species, "each state must be given for all species, but this one " \
                                                        "contained {} values".format(len(np.squeeze(state)))
        self._desired_rates = desired_rates
        self._bfc = bfc
        self._recompute = recompute
        self._set_up_system = set_up_system
        self._recompute_traj = recompute_traj
        self._fname_prefix = fname_prefix
        self._fname_postfix = fname_postfix
        self._n_species = n_species
        self._target_n_counts = target_n_counts
        self._timestep = timestep
        self._interp_degree = interp_degree
        self._trajs = []
        self._initial_states = initial_states
        self._best_alphas = {}

    @property
    def ld_derivative_config(self):
        return self._ld_derivative_config

    @ld_derivative_config.setter
    def ld_derivative_config(self, value):
        self._ld_derivative_config = value

    @property
    def interp_degree(self):
        return self._interp_degree

    @interp_degree.setter
    def interp_degree(self, value):
        self._interp_degree = value

    @property
    def timestep(self):
        return self._timestep

    @timestep.setter
    def timestep(self, value):
        self._timestep = value

    @property
    def target_n_counts(self):
        return self._target_n_counts

    @target_n_counts.setter
    def target_n_counts(self, value):
        self._target_n_counts = value

    @property
    def n_species(self):
        return self._n_species

    @n_species.setter
    def n_species(self, value):
        self._n_species = value

    @property
    def fname_postfix(self):
        return self._fname_postfix

    @fname_postfix.setter
    def fname_postfix(self, value):
        self._fname_postfix = value

    @property
    def fname_prefix(self):
        return self._fname_prefix

    @fname_prefix.setter
    def fname_prefix(self, value):
        self._fname_prefix = value

    @property
    def recompute_traj(self):
        return self._recompute_traj

    @recompute_traj.setter
    def recompute_traj(self, value):
        self._recompute_traj = value

    @property
    def set_up_system(self):
        return self._set_up_system

    @set_up_system.setter
    def set_up_system(self, value):
        self._set_up_system = value

    @property
    def recompute(self):
        return self._recompute

    @recompute.setter
    def recompute(self, value):
        self._recompute = value

    @property
    def bfc(self):
        return self._bfc

    @bfc.setter
    def bfc(self, value):
        self._bfc = value

    @property
    def desired_rates(self):
        return self._desired_rates

    @desired_rates.setter
    def desired_rates(self, value):
        self._desired_rates = value

    @property
    def best_alphas(self):
        return self._best_alphas

    @property
    def initial_states(self):
        return self._initial_states

    @initial_states.setter
    def initial_states(self, value):
        self._initial_states = value

    def get_traj_fname(self, n):
        return self._fname_prefix + "_traj_{}_".format(n) + self._fname_postfix + ".npz"

    def generate_trajectories(self, mode='gillespie', **kw):
        if mode == 'gillespie':
            for i in range(len(self._initial_states)):
                self._trajs.append(self.generate_or_load_traj_gillespie(i, **kw))
        elif mode == 'LMA':
            for i in range(len(self._initial_states)):
                self._trajs.append(self.generate_or_load_traj_lma(i, **kw))

    def obtain_serialized_gillespie_trajectories(self, desired_n_counts=6000, alphas=None, n_steps=250,
                                                 n_realizations=160, update_and_persist=False, njobs=8, atol=1e-9,
                                                 alpha_search_depth=5):
        self._trajs = []

        for n in range(len(self.initial_states)):
            traj = self.generate_or_load_traj_gillespie(n, n_steps=n_steps, n_realizations=n_realizations,
                                                        update_and_persist=update_and_persist, njobs=njobs)
            a, _ = obtain_derivative(traj, desired_n_counts=desired_n_counts, alpha=alphas, atol=atol,
                                     alpha_search_depth=alpha_search_depth, interp_degree=self.interp_degree)
            if a is None or len(a) == 0:
                if os.path.exists(self.get_traj_fname(n)):
                    t = np.load(self.get_traj_fname(n))
                    if "alpha" in t.keys():
                        a = np.copy(t['alpha'])
                    del t

            if a is not None and len(a) > 0:
                self._best_alphas[n] = a
            self._trajs.append(self.get_traj_fname(n))

    def obtain_lma_trajectories(self, target_time, alphas=None, noise_variance=0, atol=1e-9, verbose=False, njobs=8):
        self._trajs = []

        for n in range(len(self.initial_states)):
            traj = self.generate_or_load_traj_lma(n, target_time, noise_variance=noise_variance)
            _, _ = obtain_derivative(traj, desired_n_counts=self.target_n_counts, interp_degree=self.interp_degree,
                                     alpha=alphas, atol=atol, variance=noise_variance, verbose=verbose, njobs=njobs)
            self._trajs.append(self.get_traj_fname(n))

    def calculate_ld_derivatives(self, desired_n_counts=6000, alphas=None):
        for ix, traj in enumerate(self._trajs):
            a, _ = obtain_derivative(traj, desired_n_counts=desired_n_counts, alpha=alphas)
            self._best_alphas[ix] = a

    def generate_or_load_traj_gillespie(self, n, n_steps=250, n_realizations=160,
                                        update_and_persist=False, njobs=8):
        assert n < len(self._initial_states)
        init = self._initial_states[n]
        fname = self.get_traj_fname(n)
        if self._recompute_traj or not os.path.exists(fname):
            if os.path.exists(fname):
                os.remove(fname)
            _, counts = generate.generate_averaged_kmc_counts(lambda: self._set_up_system(init), n_steps,
                                                              self._timestep,
                                                              n_realizations=n_realizations, njobs=njobs)
        else:
            counts = fname
        stride = 1
        if self._target_n_counts is not None:
            if not isinstance(counts, str):
                stride = int(counts.shape[0] // self._target_n_counts)
                counts = counts[::stride]
        dt = self._timestep * stride
        traj = tools.Trajectory(counts, dt, interpolation_degree=self._interp_degree, verbose=False,
                                fname=fname, **self._ld_derivative_config)
        if update_and_persist:
            traj.update()
            traj.persist()
        return traj

    def generate_or_load_traj_lma(self, n, target_time, update_and_persist=False, noise_variance=0):
        assert n < len(self._initial_states)
        init = self._initial_states[n]
        fname = self.get_traj_fname(n)
        if self._recompute_traj or not os.path.exists(fname):
            if os.path.exists(fname):
                os.remove(fname)
            _, counts = generate.generate_continuous_counts(self._desired_rates, init, self._bfc,
                                                            self._timestep, target_time / self._timestep)
            if noise_variance > 0:
                counts = counts + np.random.normal(0.0, noise_variance, size=counts.shape)
        else:
            counts = fname
        stride = 1
        if self._target_n_counts is not None:
            if not isinstance(counts, str):
                stride = int(counts.shape[0] // self._target_n_counts)
                counts = counts[::stride]
        dt = self._timestep * stride
        traj = tools.Trajectory(counts, dt, interpolation_degree=self._interp_degree, verbose=False,
                                fname=fname, **self._ld_derivative_config)
        # system = self._set_up_system(init)
        # traj.interpolation_degree = None
        # suite = sample_tools.Suite.from_trajectory(traj, system, self._bfc, interp_degree=self._interp_degree,
        #                                            tol=tol, init_xi=np.ones_like(self._desired_rates) * .0)
        # estimator = suite.get_estimator(verbose=True, interp_degree=self._interp_degree)
        # data, expected = estimator._get_slice(None)
        # theta = estimator.get_theta(data)
        # theta = np.transpose(theta, axes=(0, 2, 1))
        # dx = theta.dot(self._desired_rates)
        # traj.dcounts_dt = dx

        traj.interpolation_degree = self._interp_degree
        if update_and_persist:
            traj.update()
            traj.persist()
        return traj

    def plot_results(self, traj, rates, title=None, outfile=None):
        from scipy.integrate import odeint
        bfc = self._bfc

        def fun(data, _):
            theta = np.array([f(data) for f in bfc.functions])
            return np.matmul(rates, theta)

        def fun_reference(data, _):
            theta = np.array([f(data) for f in bfc.functions])
            return np.matmul(self._desired_rates, theta)

        if isinstance(traj, int):
            traj = self._trajs[traj]

        if isinstance(traj, str):
            traj = tools.Trajectory(traj, self.timestep, interpolation_degree=self.interp_degree, verbose=False)
            traj.update()

        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        # f.suptitle("least squares fit for full trajectory (not well-mixed in the last time steps)")
        xs = traj.times
        num_solution = odeint(fun, traj.counts[0], xs)
        reference_soln = odeint(fun_reference, traj.counts[0], xs)
        axes = [ax1, ax2, ax3, ax4]
        labels = ["A", "B", "C", "D"]
        if title is not None:
            f.suptitle(title)
        for i in range(traj.n_species):
            axes[i].plot(xs, traj.counts[:, i], label='gillespie realization')
            axes[i].plot(xs, num_solution[:, i], label='estimated rates')
            axes[i].plot(xs, reference_soln[:, i], 'k--', label='original rates')
            axes[i].set_title("Concentration of %s particles over time" % labels[i])
            axes[i].legend()

        if outfile is not None:
            f.savefig(outfile)
        plt.show()
        return traj

    def plot_concentration_curves(self, n, fname=None):
        traj = self._trajs[n]
        if isinstance(traj, str):
            traj = tools.Trajectory(traj, self.timestep, interpolation_degree=self.interp_degree, verbose=False)
        system = self._set_up_system(self._initial_states[n])
        config = system.get_trajectory_config()
        estimated = sample_tools.Suite.estimated_behavior(self._desired_rates, self._bfc, traj.counts[0], traj.times)

        fig, ax1 = plt.subplots(nrows=1, ncols=1)

        fig.suptitle('Training trajectory')
        ax1.set_xlabel('time')
        ax1.set_ylabel('concentration')
        for t in config.types.keys():
            type_id = config.types[t]
            ax1.plot(traj.times, traj.counts[:, type_id], label="concentration " + t)
            ax1.plot(traj.times, estimated[:, type_id], "k--",
                     label=None if type_id != 3 else "law of mass action solution")
        ax1.legend(loc="upper right")
        if fname is not None:
            fig.savefig(fname)

    def get_lsq_fname(self, n):
        return self._fname_prefix + "_lsq_{}_".format(n) + self._fname_postfix + ".npz"

    def least_squares(self, n, tol=1e-12, initial_guess=None):
        fname = self.get_lsq_fname(n)
        if self._recompute or not os.path.exists(fname):
            system = self._set_up_system(self._initial_states[n])
            traj = self._trajs[n]
            if isinstance(traj, str):
                traj = tools.Trajectory(traj, self.timestep, interpolation_degree=self.interp_degree, verbose=False)
            if initial_guess is None:
                initial_guess = np.zeros_like(self._desired_rates)
            suite = sample_tools.Suite.from_trajectory(traj, system, self._bfc, interp_degree=self._interp_degree,
                                                       tol=tol,
                                                       init_xi=initial_guess)
            estimator = suite.get_estimator(verbose=True, interp_degree=self._interp_degree)
            estimator.fit(None)
            if estimator.success_:
                rates = estimator.coefficients_
                np.save(fname, rates)
            else:
                raise ValueError("Didn't converge :-/")
        else:
            rates = np.load(fname)
        return rates

    def get_cv_fname(self, n_train):
        return self._fname_prefix + "_cv_train_{}_".format(n_train) + self._fname_postfix + ".npy"

    def elastic_net(self, train_n, alphas, l1_ratios, test_n=None, initial_guess=None, tol=1e-16, njobs=8):
        if test_n is None:
            test_n = [self._trajs[i] for i in range(len(self._trajs)) if i != train_n]
        else:
            test_n = [self._trajs[test_n]]
        if initial_guess is None:
            initial_guess = np.zeros_like(self._desired_rates)
        fname = self.get_cv_fname(n_train=train_n)
        traintraj = self._trajs[train_n]
        if isinstance(traintraj, str):
            traintraj = tools.Trajectory(traintraj, self.timestep, interpolation_degree=self.interp_degree,
                                         verbose=False)
            traintraj.update()
        cv = rlas.CV(traintraj, self._bfc, alphas, l1_ratios, 5, initial_guess,
                     test_traj=test_n, maxiter=300000, rescale=False, tol=tol, n_jobs=njobs)
        if self._recompute or not os.path.exists(fname):
            cv.fit_cross_trajs()
            np.save(fname, cv.result)
        else:
            cv.result = np.load(fname)
        return cv

    def get_solve_fname(self, n):
        return self.fname_prefix + "_solution_{}_".format(n)+self.fname_postfix+".npy"

    def solve(self, n, alpha, l1_ratio, tol=1e-12, recompute=False):

        if not recompute:
            if not self.recompute and os.path.exists(self.get_solve_fname(n)):
                return np.load(self.get_solve_fname(n))

        system = self._set_up_system(self.initial_states[n])
        traj = self._trajs[n]
        if isinstance(traj, str):
            traj = tools.Trajectory(traj, self.timestep, interpolation_degree=self.interp_degree, verbose=False)
            traj.update()

        optsuite = sample_tools.Suite.from_trajectory(traj, system, self._bfc, interp_degree=self.interp_degree,
                                                      tol=tol, alpha=alpha, l1_ratio=l1_ratio,
                                                      init_xi=np.zeros_like(self.desired_rates))
        estimator = optsuite.get_estimator(verbose=True, interp_degree=self.interp_degree)
        estimator.fit(None)
        if estimator.success_:
            rates = estimator.coefficients_
            np.save(self.get_solve_fname(n), rates)
            return rates
        else:
            raise ValueError('*_*')

    def plot_derivatives(self, traj_n, n_points=None):
        traj = self._trajs[traj_n]
        if isinstance(traj, str):
            traj = tools.Trajectory(traj, self.timestep, interpolation_degree=self.interp_degree, verbose=False)
        init = self._initial_states[traj_n]
        system = self._set_up_system(init)
        suite = sample_tools.Suite.from_trajectory(traj, system, self._bfc, interp_degree=self._interp_degree,
                                                   tol=1e-12, init_xi=np.ones_like(self._desired_rates) * .0)
        if n_points is not None:
            stride = max(len(traj.times) // n_points, 1)
        estimator = suite.get_estimator(verbose=True, interp_degree=self._interp_degree)
        data, expected = estimator._get_slice(None)
        theta = estimator.get_theta(data)
        theta = np.transpose(theta, axes=(0, 2, 1))
        dx = theta.dot(self._desired_rates)

        for s in range(traj.n_species):
            if n_points is None:
                plt.plot(traj.times, dx[:, s], 'k--')
                plt.plot(traj.times, traj.dcounts_dt[:, s], label="dx for species {}".format(s))
            else:
                plt.plot(traj.times[::stride], dx[:, s][::stride], 'k--')
                plt.plot(traj.times[::stride], traj.dcounts_dt[:, s][::stride], label="dx for species {}".format(s))
        plt.legend()
        plt.show()
