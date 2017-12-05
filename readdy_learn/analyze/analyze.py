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


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def obtain_derivative(traj, desired_n_counts=6000, alpha=1000, atol=1e-10, tol=1e-10, maxit=1000, alpha_search_depth=5,
                      interp_degree='regularized_derivative', variance=None, verbose=False, x0=None,
                      species=None, override=False, subdivisions=None, reuse_deriv=True):
    if species is None:
        species = [i for i in range(traj.n_species)]
    species = np.array(species).squeeze()
    print("obtaining derivative for species {}".format(species))
    if traj.dcounts_dt is None or override:
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

            x0 = np.asarray(x0).squeeze()
            used_alphas = []
            for s in species if species.ndim > 0 else [species]:
                if int(s) in traj.separate_derivs.keys() and not override:
                    print("skipping species {} as it already has derivatives".format(s))
                    continue
                ys = strided_counts[:, s]
                kw = {'maxit': maxit, 'linalg_solver_maxit': 50000, 'tol': tol, 'atol': atol, 'rtol': None,
                      'solver': 'spsolve', 'verbose': False, 'show_progress': verbose}
                if isinstance(alpha, np.ndarray):
                    if len(alpha) > 1:
                        if subdivisions is None:
                            best_alpha, ld, scores = deriv.best_tv_derivative(ys, strided_times, alpha,
                                                                              n_iters=alpha_search_depth,
                                                                              variance=variance,
                                                                              reuse_deriv=reuse_deriv,
                                                                              x0=x0[s], **kw)
                        else:
                            assert isinstance(subdivisions, int)
                            interp = deriv.interpolate(strided_times, ys)
                            subys = list(split(ys, subdivisions))
                            subtimes = list(split(strided_times, subdivisions))
                            subinterp = list(split(interp, subdivisions))
                            ld = None
                            subalphs = []
                            for i in range(subdivisions):
                                print("----------------------------- subdiv {} ----------------------------".format(i))
                                init = x0[s] if ld is None else subinterp[i][0]
                                print("got initial value {}".format(init))
                                subalph, subld, subscores = deriv.best_tv_derivative(subys[i], subtimes[i], alpha,
                                                                                     n_iters=alpha_search_depth,
                                                                                     variance=variance,
                                                                                     reuse_deriv=reuse_deriv,
                                                                                     x0=init, **kw)
                                print("found alpha={}".format(subalph))
                                if ld is not None:
                                    ld = np.append(ld, subld)
                                else:
                                    ld = subld
                                subalphs.append(subalph)
                            best_alpha = subalphs
                    else:
                        alpha = alpha[0]
                        ld = deriv.tv_derivative(ys, strided_times, alpha=alpha, **kw)
                        best_alpha = alpha
                else:
                    ld = deriv.tv_derivative(ys, strided_times, alpha=alpha, **kw)
                    best_alpha = alpha
                # linearly interpolate to the full time range
                integrated_ld = deriv.integrate.cumtrapz(ld, x=strided_times, initial=0) + \
                                x0[s] if x0 is not None else ys[0]
                if variance is not None:
                    var = variance
                else:
                    var, ff = estimate_noise_variance(strided_times, ys)
                print("MSE =", deriv.mse(integrated_ld, ys), "noise variance =", var)
                traj.separate_derivs[int(s)] = np.interp(traj.times, strided_times, ld)

                used_alphas.append(best_alpha)
            traj.interpolation_degree = interp_degree
            traj.persist(alpha=used_alphas)
            return used_alphas, traj
        else:
            traj.update()
            traj.persist()
            return [], traj
    else:
        print("traj already contains derivatives and override is {}, skip this".format(override))
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

    def obtain_lma_trajectories(self, target_time, alphas=None, noise_variance=0, atol=1e-9, tol=1e-12, verbose=False,
                                maxit=2000, search_depth=10, selection=None, species=None, override=False,
                                subdivisions=None, reuse_deriv=True):
        if species is None:
            species = [i for i in range(self.n_species)]
        self._trajs = [None for _ in range(len(self.initial_states))]
        for n in range(len(self.initial_states)):
            if selection is None or n in selection:
                traj = self.generate_or_load_traj_lma(n, target_time, noise_variance=noise_variance)
                _, _ = obtain_derivative(traj, desired_n_counts=self.target_n_counts, interp_degree=self.interp_degree,
                                         alpha=alphas, atol=atol, variance=noise_variance, verbose=verbose, tol=tol,
                                         maxit=maxit, alpha_search_depth=search_depth, x0=self.initial_states[n],
                                         species=species, override=override, subdivisions=subdivisions,
                                         reuse_deriv=reuse_deriv)
                self._trajs[n] = self.get_traj_fname(n)

    def calculate_ld_derivatives(self, desired_n_counts=6000, alphas=None, maxit=10):
        for ix, traj in enumerate(self._trajs):
            a, _ = obtain_derivative(traj, desired_n_counts=desired_n_counts, alpha=alphas, maxit=maxit)
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
                counts = counts + np.random.normal(0.0, np.sqrt(noise_variance), size=counts.shape)
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

    def plot_concentration_curves(self, n, fname=None, species=None, plot_estimated=True):
        if species is None:
            species = [i for i in range(self.n_species)]
        traj = self._trajs[n]
        if isinstance(traj, str):
            traj = tools.Trajectory(traj, self.timestep, interpolation_degree=self.interp_degree, verbose=False)
        system = self._set_up_system(self._initial_states[n])
        config = system.get_trajectory_config()
        estimated = sample_tools.Suite.estimated_behavior(self._desired_rates, self._bfc,
                                                          self.initial_states[n].squeeze(), traj.times)

        fig, ax1 = plt.subplots(nrows=1, ncols=1)

        fig.suptitle('Training trajectory')
        ax1.set_xlabel('time')
        ax1.set_ylabel('concentration')
        for t in config.types.keys():
            type_id = config.types[t]
            if type_id in species:
                ax1.plot(traj.times, traj.counts[:, type_id], label="concentration " + t)
                ax1.plot(traj.times, estimated[:, type_id], "k--",
                         label=None if type_id != 3 else "law of mass action solution")
                if plot_estimated:
                    integrated_ld = deriv.integrate.cumtrapz(traj.separate_derivs[type_id], x=traj.times, initial=0) \
                                    + self.initial_states[n].squeeze()[type_id]
                    ax1.plot(traj.times, integrated_ld, "r--", label="integrated derivative")
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

    def get_traj(self, n):
        traj = self._trajs[n]
        if isinstance(traj, str):
            traj = tools.Trajectory(traj, self.timestep, interpolation_degree=self.interp_degree,
                                    verbose=False)
            traj.update()
        return traj

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
        return self.fname_prefix + "_solution_{}_".format(n) + self.fname_postfix + ".npy"

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

    def plot_derivatives(self, traj_n, n_points=None, species=None):
        if species is None:
            species = [i for i in range(self.n_species)]
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

        for s in species:
            if n_points is None:
                plt.plot(traj.times, dx[:, s], 'k--')
                plt.plot(traj.times, traj.separate_derivs[s], label="dx for species {}".format(s))
            else:
                plt.plot(traj.times[::stride], dx[:, s][::stride], 'k--')
                plt.plot(traj.times[::stride], traj.separate_derivs[s][::stride], label="dx for species {}".format(s))
        plt.legend()
        plt.show()


def plot_cv_results(cv, mainscore=0, best_params_ix_l1=1.):
    xs = {}
    ys = {}
    allys = {}
    for r in cv.result:
        l1_ratio = r['l1_ratio']
        if len(r['scores']) > 0:
            if l1_ratio in xs.keys():
                xs[l1_ratio].append(r['alpha'])
                ys[l1_ratio].append(r['scores'][mainscore])
                allys[l1_ratio].append(r['scores'])
            else:
                xs[l1_ratio] = [r['alpha']]
                ys[l1_ratio] = [r['scores'][mainscore]]
                allys[l1_ratio] = [r['scores']]
    f, ax = plt.subplots(figsize=(20, 20))
    for l1_ratio in xs.keys():
        l1xs = np.array(xs[l1_ratio])
        l1ys = np.array(ys[l1_ratio])
        l1allys = np.array([np.array(arr) for arr in allys[l1_ratio]]).T
        sorts = np.argsort(l1xs)
        l1xs = l1xs[sorts]
        l1ys = l1ys[sorts]

        l1allys = [arr[sorts] for arr in l1allys]
        if l1_ratio == best_params_ix_l1 or best_params_ix_l1 is None:
            ax.plot(l1xs, -l1ys, label='score l1={}'.format(l1_ratio))

            for ix, _ys in enumerate(l1allys):
                if np.argmin(-_ys) != 0:
                    # print("found one: {} with argmin {}".format(ix, np.argmin(_ys)))
                    pass
                # ax.plot(l1xs, -_ys, label='test set {}'.format(ix))
                pass
    f.suptitle('Cross-validation scores')
    ax.set_ylabel('score')
    ax.set_xlabel('$\\alpha$')
    plt.legend()
    plt.show()


def plot_rates_bar(desired_rates, estimated_rates, color1='blue', color2='green'):
    assert len(desired_rates) == len(estimated_rates)
    N = len(desired_rates)
    ind = np.arange(N)
    width = .35
    fig, ax = plt.subplots()
    bar1 = ax.bar(ind, desired_rates, width, color=color1)
    bar2 = ax.bar(ind + width, estimated_rates, width, color=color2)
    ax.set_xticks(ind + width / 2)
    ax.legend((bar1[0], bar2[0]), ('Desired', 'Estimated'))
    ax.set_xticklabels(["{}".format(i) for i in ind])
    plt.show()


def best_params(cv, test_traj=None):
    current_best_score = -1
    alpha = -1
    l1_ratio = -1

    for r in cv.result:
        if len(r['scores']) > 0:
            if test_traj is None:
                current_score = np.mean(r['scores'])
            else:
                current_score = r['scores'][test_traj]
            if current_best_score >= 0:
                if -current_score < current_best_score:
                    current_best_score = -current_score
                    alpha = r['alpha']
                    l1_ratio = r['l1_ratio']
            else:
                current_best_score = -current_score
                alpha = r['alpha']
                l1_ratio = r['l1_ratio']
    return alpha, l1_ratio, current_best_score


def do_the_cv(analysis, n, alphas, l1_ratios, tol=1e-12, solvetol=1e-15, plot_cv_for=None, best_params_ix=None,
              best_params_ix_l1=None, cutoff=1e-8, recompute=False):
    cv_n = analysis.elastic_net(n, alphas, l1_ratios, tol=tol)
    if plot_cv_for is not None:
        plot_cv_results(cv_n, mainscore=plot_cv_for, best_params_ix_l1=best_params_ix_l1)
    alpha, l1_ratio, score = best_params(cv_n, best_params_ix)
    print("params: alpha={}, l1={} with corresponding score {}".format(alpha, l1_ratio, score))
    rates = analysis.solve(n, alpha, l1_ratio, tol=solvetol, recompute=recompute)
    rates[np.where(rates <= cutoff)] = 0
    return rates
