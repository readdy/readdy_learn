import numpy as _np
import pynumtools.kmc as _kmc
import readdy_learn.analyze.basis as _basis
import matplotlib.pyplot as plt
from pathos.multiprocessing import Pool as _Pool
import readdy_learn.analyze.progress as _pr
import readdy_learn.analyze.interface as _interface
import readdy_learn.analyze.analyze as _ana
from pynumtools.finite_differences import fd_coefficients as _fd_coeffs
from pynumtools.util import sliding_window as _sliding_window

DEFAULT_DESIRED_RATES = _np.array([
    1.8,  # DA -> DA + MA, transcription A
    2.1,  # MA -> MA + A, translation A
    1.3,  # MA -> 0, decay
    1.5,  # A -> 0, decay
    2.2,  # DB -> DB + MB, transcription B
    2.0,  # MB -> MB + B, translation B
    2.0,  # MB -> 0, decay
    2.5,  # B -> 0, decay
    3.2,  # DC -> DC + MC, transcription C
    3.0,  # MC -> MC + C, translation C
    2.3,  # MC -> 0, decay
    2.5,  # C -> 0, decay
    # self regulation
    0.,  # MA + A -> A, A regulates A
    0.,  # MB + B -> B, B regulates B
    0.,  # MC + C -> C, C regulates C
    # cyclic forward
    0.,  # MB + A -> A, A regulates B
    0.,  # MC + B -> B, B regulates C
    0.,  # MA + C -> C, C regulates A
    # cyclic backward
    6.,  # MC + A -> A, A regulates C
    4.,  # MB + C -> C, C regulates B
    3.,  # MA + B -> B, B regulates A
    # nonsense reactions, mRNA eats protein self
    0., 0., 0.,
    # nonsense reactions, mRNA eats protein cyclic forward
    0., 0., 0.,
    # nonsense reactions, mRNA eats protein  cyclic backward
    0., 0., 0.,
    # nonsense reactions, protein eats protein self
    0., 0., 0.,
    # nonsense reactions, protein eats protein cyclic forward
    0., 0., 0.,
    # nonsense reactions, protein eats protein cyclic backward
    0., 0., 0.,
    # nonsense reactions, protein becomes protein cyclic forward
    0., 0., 0.,
    # nonsense reactions, protein becomes protein cyclic backward
    0., 0., 0.,
])


class RegulationNetwork(_interface.AnalysisObjectGenerator):

    def __init__(self):
        # species DA  MA  A  DB  MB  B  DC  MC  C
        # ids     0   1   2  3   4   5  6   7   8
        self.n_species = 9
        self.species_names = ["DA", "MA", "A", "DB", "MB", "B", "DC", "MC", "C"]
        self._desired_rates = self.get_desired_rates()

        initial_states = [
            [1, 0, 0, 1, 0, 0, 1, 0, 0], [1, 2, 0, 1, 0, 3, 1, 0, 0], [1, 1, 2, 1, 0, 2.5, 1, 0, 2],
            [1, 1, 2, 1, 0, 0, 1, 3, 0],
            [1, 2, 0, 1, 0, 3, 1, 0, 1], [1, 0, 2, 1, 0, 2.5, 1, 0.5, 0]
        ]
        self._initial_states = [_np.array([arr]) for arr in initial_states]

        self.ld_derivative_config = {
            'ld_derivative_atol': 1e-4,
            'ld_derivative_rtol': None,
            'ld_derivative_linalg_solver_maxit': 100000,
            'ld_derivative_alpha': 1e-1,
            'ld_derivative_solver': 'spsolve',
            'ld_derivative_linalg_solver_tol': 1e-10,
            'ld_derivative_use_preconditioner': False
        }

        self._noise_variance = 1e-2
        self._target_time = 3.
        self.realisations = 60
        self.timestep = 1e-3

    def get_desired_rates(self):
        return DEFAULT_DESIRED_RATES

    def set_up_system(self, init_state):
        sys = _kmc.ReactionDiffusionSystem(diffusivity=self.n_species * [[[0.]]], n_species=self.n_species, n_boxes=1,
                                           init_state=init_state, species_names=self.species_names)
        # usual stuff A
        sys.add_fission("DA", "DA", "MA", _np.array([self.desired_rates[0]]))  # DA -> DA + MA transcription
        sys.add_fission("MA", "MA", "A", _np.array([self.desired_rates[1]]))  # MA -> MA + A translation
        sys.add_decay("MA", _np.array([self.desired_rates[2]]))  # MA -> 0 mRNA A decay
        sys.add_decay("A", _np.array([self.desired_rates[3]]))  # A -> 0 protein decay
        # usual stuff B
        sys.add_fission("DB", "DB", "MB", _np.array([self.desired_rates[4]]))  # DB -> DB + MB transcription
        sys.add_fission("MB", "MB", "B", _np.array([self.desired_rates[5]]))  # MB -> MB + B translation
        sys.add_decay("MB", _np.array([self.desired_rates[6]]))  # MB -> 0 mRNA B decay
        sys.add_decay("B", _np.array([self.desired_rates[7]]))  # B -> 0 protein decay
        # usual stuff C
        sys.add_fission("DC", "DC", "MC", _np.array([self.desired_rates[8]]))  # DC -> DC + MC transcription
        sys.add_fission("MC", "MC", "C", _np.array([self.desired_rates[9]]))  # MC -> MC + C translation
        sys.add_decay("MC", _np.array([self.desired_rates[10]]))  # MC -> 0 mRNA C decay
        sys.add_decay("C", _np.array([self.desired_rates[11]]))  # C -> 0 protein decay

        # regulation: only the real ones show up here
        # self regulation
        # sys.add_fusion("MA", "A", "A", _np.array([desired_rates[12]]))  # MA + A -> A, A regulates A
        # sys.add_fusion("MB", "B", "B", _np.array([desired_rates[13]]))  # MB + B -> B, B regulates B
        # sys.add_fusion("MC", "C", "C", _np.array([desired_rates[14]]))  # MC + C -> C, C regulates C
        # cyclic forward
        # sys.add_fusion("MB", "A", "A", _np.array([desired_rates[15]])) # MB + A -> A, A regulates B
        # sys.add_fusion("MC", "B", "B", _np.array([desired_rates[16]])) # MC + B -> B, B regulates C
        # sys.add_fusion("MA", "C", "C", _np.array([desired_rates[17]])) # MA + C -> C, C regulates A
        # cyclic backward
        sys.add_fusion("MC", "A", "A", _np.array([self.desired_rates[18]]))  # MC + A -> A, A regulates C
        sys.add_fusion("MB", "C", "C", _np.array([self.desired_rates[19]]))  # MB + C -> C, C regulates B
        sys.add_fusion("MA", "B", "B", _np.array([self.desired_rates[20]]))  # MA + B -> B, B regulates A

        return sys

    def get_bfc(self):
        # species DA  MA  A  DB  MB  B  DC  MC  C
        # ids     0   1   2  3   4   5  6   7   8
        bfc = _basis.BasisFunctionConfiguration(self.n_species)
        # usual stuff A
        bfc.add_fission(0, 0, 1)  # 0   DA -> DA + MA, transcription A
        bfc.add_fission(1, 1, 2)  # 1   MA -> MA + A, translation A
        bfc.add_decay(1)  # 2   MA -> 0, decay
        bfc.add_decay(2)  # 3   A -> 0, decay
        # usual stuff B
        bfc.add_fission(3, 3, 4)  # 4   DB -> DB + MB, transcription B
        bfc.add_fission(4, 4, 5)  # 5   MB -> MB + B, translation B
        bfc.add_decay(4)  # 6   MB -> 0, decay
        bfc.add_decay(5)  # 7   B -> 0, decay
        # usual stuff C
        bfc.add_fission(6, 6, 7)  # 8   DC -> DC + MC, transcription C
        bfc.add_fission(7, 7, 8)  # 9   MC -> MC + C, translation C
        bfc.add_decay(7)  # 10  MC -> 0, decay
        bfc.add_decay(8)  # 11  C -> 0, decay

        # all possible regulations
        # self regulation
        bfc.add_fusion(1, 2, 2)  # 12  MA + A -> A, A regulates A
        bfc.add_fusion(4, 5, 5)  # 13  MB + B -> B, B regulates B
        bfc.add_fusion(7, 8, 8)  # 14  MC + C -> C, C regulates C
        # cyclic forward
        bfc.add_fusion(4, 2, 2)  # 15  MB + A -> A, A regulates B
        bfc.add_fusion(7, 5, 5)  # 16  MC + B -> B, B regulates C
        bfc.add_fusion(1, 8, 8)  # 17  MA + C -> C, C regulates A
        # cyclic backward
        bfc.add_fusion(7, 2, 2)  # 18  MC + A -> A, A regulates C
        bfc.add_fusion(4, 8, 8)  # 19  MB + C -> C, C regulates B
        bfc.add_fusion(1, 5, 5)  # 20  MA + B -> B, B regulates A

        # thrown these out due to being identical to decay
        # nonsense reactions, DNA eats mRNA self
        # bfc.add_fusion(1, 0, 0) # 21 MA + DA -> DA
        # bfc.add_fusion(4, 3, 3) # 22 MB + DB -> DB
        # bfc.add_fusion(7, 6, 6) # 23 MC + DC -> DC

        # nonsense reactions, DNA eats mRNA cyclic forward
        # bfc.add_fusion(4, 0, 0) # 24 MB + DA -> DA
        # bfc.add_fusion(7, 3, 3) # 25 MC + DB -> DB
        # bfc.add_fusion(1, 6, 6) # 26 MA + DC -> DC

        # nonsense reactions, DNA eats mRNA cyclic backward
        # bfc.add_fusion(7, 0, 0) # 27 MC + DA -> DA
        # bfc.add_fusion(4, 6, 6) # 28 MB + DC -> DC
        # bfc.add_fusion(1, 3, 3) # 29 MA + DB -> DB

        # nonsense reactions, mRNA eats protein self
        bfc.add_fusion(1, 2, 1)  # 21 MA + A -> MA
        bfc.add_fusion(4, 5, 4)  # 22 MB + B -> MB
        bfc.add_fusion(7, 8, 8)  # 23 MC + C -> MC

        # nonsense reactions, mRNA eats protein cyclic forward
        bfc.add_fusion(1, 5, 1)  # 24 MA + B -> MA
        bfc.add_fusion(4, 8, 4)  # 25 MB + C -> MB
        bfc.add_fusion(7, 2, 7)  # 26 MC + A -> MC

        # nonsense reactions, mRNA eats protein  cyclic backward
        bfc.add_fusion(1, 8, 1)  # 27 MA + C -> MA
        bfc.add_fusion(4, 2, 4)  # 28 MB + A -> MB
        bfc.add_fusion(7, 4, 7)  # 29 MC + B -> MC

        # nonsense reactions, protein eats protein self
        bfc.add_fusion(2, 2, 2)  # 30 A + A -> A
        bfc.add_fusion(5, 5, 5)  # 31 B + B -> B
        bfc.add_fusion(8, 8, 8)  # 32 C + C -> C

        # nonsense reactions, protein eats protein cyclic forward
        bfc.add_fusion(5, 2, 2)  # 30 B + A -> A
        bfc.add_fusion(8, 5, 5)  # 31 C + B -> B
        bfc.add_fusion(2, 8, 8)  # 32 A + C -> C

        # nonsense reactions, protein eats protein cyclic backward
        bfc.add_fusion(8, 2, 2)  # 33 C + A -> A
        bfc.add_fusion(5, 8, 8)  # 34 B + C -> C
        bfc.add_fusion(2, 5, 5)  # 35 A + B -> B

        # nonsense reactions, protein becomes protein cyclic forward
        bfc.add_conversion(2, 5)  # 36 A -> B
        bfc.add_conversion(5, 8)  # 37 B -> C
        bfc.add_conversion(8, 2)  # 38 C -> A

        # nonsense reactions, protein becomes protein cyclic backward
        bfc.add_conversion(2, 8)  # 39 A -> C
        bfc.add_conversion(8, 5)  # 40 C -> B
        bfc.add_conversion(5, 2)  # 41 B -> A

        assert bfc.n_basis_functions == len(self.desired_rates), \
            "got {} basis functions but only {} desired rates".format(bfc.n_basis_functions, len(self.desired_rates))
        return bfc

    def plot_concentrations(self, analysis, traj_n):
        traj_number = traj_n
        pfs = [
            lambda t=traj_number: analysis.plot_derivatives(t, species=[0]),
            lambda t=traj_number: analysis.plot_concentration_curves(t, species=[0]),
            lambda t=traj_number: analysis.plot_derivatives(t, species=[3]),
            lambda t=traj_number: analysis.plot_concentration_curves(t, species=[3]),
            lambda t=traj_number: analysis.plot_derivatives(t, species=[6]),
            lambda t=traj_number: analysis.plot_concentration_curves(t, species=[6]),
            lambda t=traj_number: analysis.plot_derivatives(t, species=[1, 2]),
            lambda t=traj_number: analysis.plot_concentration_curves(t, species=[1, 2]),
            lambda t=traj_number: analysis.plot_derivatives(t, species=[4, 5]),
            lambda t=traj_number: analysis.plot_concentration_curves(t, species=[4, 5]),
            lambda t=traj_number: analysis.plot_derivatives(t, species=[7, 8]),
            lambda t=traj_number: analysis.plot_concentration_curves(t, species=[7, 8])
        ]
        analysis.plot_overview(pfs, n_cols=2, n_rows=6, size_factor=1.3)
        plt.show()

    def generate_analysis_object(self, fname_prefix=None, fname_postfix=None) -> _ana.ReactionAnalysis:
        if fname_prefix is None:
            fname_prefix = "regulation_network"
        if fname_postfix is None:
            fname_postfix = ""
        analysis = _ana.ReactionAnalysis(self.get_bfc(), self.desired_rates, self.initial_states, self.set_up_system,
                                         fname_prefix=fname_prefix, fname_postfix=fname_postfix,
                                         n_species=self.n_species, timestep=self.timestep,
                                         ld_derivative_config=self.ld_derivative_config, recompute_traj=False,
                                         species_names=self.species_names)
        return analysis

    def compute_tv_derivatives(self, analysis: _ana.ReactionAnalysis, alphas: list = _np.linspace(1e-7, 1e-1, num=10),
                               alpha_search_depth: int = 3, atol: float = 1e-10):
        for t in range(len(self.initial_states)):
            traj = analysis.get_traj(t)
            _ana.obtain_derivative(traj, alpha=alphas, atol=atol, alpha_search_depth=alpha_search_depth, override=True,
                                   x0=self.initial_states[t])

    def compute_gradient_derivatives(self, analysis: _ana.ReactionAnalysis, persist: bool = True, wwidth=None):
        for t in range(len(self.initial_states)):
            traj = analysis.get_traj(t)
            # for sp in [0, 3, 6]:
            #    dx = _np.zeros_like(traj.counts[:, sp])
            #    print("species {} dx.shape {}".format(sp, dx.shape))
            #    traj.separate_derivs[sp] = dx
            times = traj.times

            for sp in [1, 2, 4, 5, 7, 8, 0, 3, 6]:
                x = traj.counts[:, sp]
                dt = traj.time_step

                indices = 1 + _np.where(x[:-1] != x[1:])[0]
                indices = _np.insert(indices, 0, 0)

                if wwidth is not None:
                    if len(_np.atleast_1d(indices.squeeze())) == 1:
                        indices = _np.arange(2 * wwidth + 1, dtype=int)

                    unique_times, unique_counts = times[indices].squeeze(), x[indices].squeeze()
                    unique_deriv = _np.empty_like(unique_counts)

                    for ix, (wx, wy) in enumerate(
                            zip(_sliding_window(_np.atleast_1d(unique_times), width=wwidth, fixed_width=False),
                                _sliding_window(_np.atleast_1d(unique_counts), width=wwidth, fixed_width=False))):
                        x = unique_times[ix]
                        coeff = _fd_coeffs(x, wx, k=1)
                        unique_deriv[ix] = coeff.dot(wy)

                    interpolated = _np.interp(times, unique_times, unique_deriv)
                    traj.separate_derivs[sp] = interpolated
                else:
                    interpolated = _np.interp(times, times[indices], x[indices])
                    interpolated = _np.gradient(interpolated) / dt

                    traj.separate_derivs[sp] = interpolated

            if persist:
                traj.persist()

    def interpolate_counts(self, analysis: _ana.ReactionAnalysis, persist: bool = True):
        for t in range(len(self.initial_states)):
            traj = analysis.get_traj(t)
            times = traj.times

            for sp in [1, 2, 4, 5, 7, 8, 0, 3, 6]:
                x = traj.counts[:, sp]

                indices = 1 + _np.where(x[:-1] != x[1:])[0]
                indices = _np.insert(indices, 0, 0)

                interpolated = _np.interp(times, times[indices], x[indices])

                traj.counts[:, sp] = interpolated

            if persist:
                traj.persist()

    @property
    def desired_rates(self):
        return self._desired_rates

    @desired_rates.setter
    def desired_rates(self, value):
        self._desired_rates = value

    @property
    def initial_states(self):
        return self._initial_states

    @initial_states.setter
    def initial_states(self, value):
        self._initial_states = value

    @property
    def target_time(self):
        return self._target_time

    @target_time.setter
    def target_time(self, value):
        self._target_time = value

    @property
    def noise_variance(self):
        return self._noise_variance

    @noise_variance.setter
    def noise_variance(self, value):
        self._noise_variance = value


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = _np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = _np.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = _np.convolve(w / w.sum(), s, mode='valid')
    return y


def sample_along_alpha(regulation_network, alphas, samples_per_alpha=1, njobs=8, tol=1e-16, verbose=False):
    if not isinstance(alphas, (list, tuple, _np.ndarray)):
        alphas = [alphas]

    result = {}

    progress = _pr.Progress(n=samples_per_alpha * len(alphas), label="sample for alphas")

    def worker(args):
        alpha = args[0]
        analysis = regulation_network.generate_analysis_object(fname_prefix='case_1', fname_postfix='0')
        for i in range(len(regulation_network.initial_states)):
            analysis.generate_or_load_traj_lma(i, regulation_network.target_time,
                                               noise_variance=regulation_network.noise_variance,
                                               realizations=regulation_network.realisations)
        regulation_network.compute_gradient_derivatives(analysis, persist=False)
        reg_rates = analysis.solve(0, alpha=alpha, l1_ratio=1., tol=tol, recompute=True, verbose=verbose, persist=False)
        return alpha, reg_rates

    args = [(alpha,) for _ in range(samples_per_alpha) for alpha in alphas]
    with _Pool(processes=njobs) as p:
        for x in p.imap_unordered(worker, args):
            alpha, rates = x
            if alpha in result.keys():
                result[alpha].append(rates)
            else:
                result[alpha] = [rates]
            progress.increase()
    progress.finish()

    return result


def sample_lsq_rates(realizations, base_variance=.1, samples_per_variance=8, njobs=8, timestep=6e-3):
    if not isinstance(realizations, (list, tuple, _np.ndarray)):
        realizations = [realizations]

    def get_regulation_network(n):
        regulation_network = RegulationNetwork()
        regulation_network.timestep = timestep
        regulation_network.realisations = 1.
        regulation_network.noise_variance = base_variance / n
        regulation_network.initial_states = [regulation_network.initial_states[1]]
        return regulation_network

    def get_analysis_object(regulation_network, prefix='point_1', postfix='n_{}'):
        return regulation_network.generate_analysis_object(
            fname_prefix=prefix, fname_postfix=postfix.format(regulation_network.realisations))

    def do_for_n_realizations(n, prefix='point_1', postfix='n_{}', verbose=False):
        separator = "-" * 15
        if verbose:
            print("{} n={} {}".format(separator, n, separator))
        regulation_network = get_regulation_network(n)
        analysis = get_analysis_object(regulation_network, prefix, postfix)

        for i in range(len(regulation_network.initial_states)):
            analysis.generate_or_load_traj_lma(i, regulation_network.target_time,
                                               noise_variance=regulation_network.noise_variance,
                                               realizations=regulation_network.realisations)
        regulation_network.compute_gradient_derivatives(analysis, persist=False)
        if verbose:
            print("noise variance: {}".format(regulation_network.noise_variance))
            print("target time: {}".format(regulation_network.target_time))
            print("lma realizations: {}".format(regulation_network.realisations))
            print("timestep: {}".format(regulation_network.timestep))
            print("initial states:")
            for init in regulation_network.initial_states:
                print("\t{}".format(init))
        try:
            lsq_rates = analysis.least_squares([0], tol=1e-16, recompute=True, persist=False, verbose=False)
        except ValueError:
            try:
                lsq_rates = analysis.least_squares([0], tol=1e-15, recompute=True, persist=False, verbose=False)
            except ValueError:
                try:
                    lsq_rates = analysis.least_squares([0], tol=1e-14, recompute=True, persist=False, verbose=False)
                except ValueError:
                    try:
                        lsq_rates = analysis.least_squares([0], tol=1e-13, recompute=True, persist=False, verbose=False)
                    except ValueError:
                        raise ValueError("This takes the cake")
        l2_err = analysis.compute_L2_error(0, lsq_rates)
        if verbose:
            print("|LMA-LMA_est|_2 = {}".format(l2_err))
        return {
            'lsq_rates': lsq_rates,
            'l2_err': l2_err,
            'prefix': prefix,
            'postfix': postfix.format(n),
            'n': n
        }

    import time

    def worker(args):
        n, r = args
        seed = int(time.time()) + n * samples_per_variance + r
        seed = seed % (2 ** 32 - 1)
        _np.random.seed(seed)
        return do_for_n_realizations(n)

    from collections import defaultdict
    progress = _pr.Progress(n=len(realizations) * samples_per_variance, label="sample L2 error for variances")
    result = {}
    with _Pool(processes=njobs) as p:
        for n in realizations:
            params = [(n, r) for r in range(samples_per_variance)]
            res = defaultdict(list)
            for r in p.imap(worker, params, 1):
                res['lsq_rates'].append(r['lsq_rates'])
                res['l2_err'].append(r['l2_err'])
                res['prefix'] = r['prefix']
                res['postfix'] = r['postfix']
                res['n'] = n
                progress.increase()
            result[n] = res
    progress.finish()
    return result
