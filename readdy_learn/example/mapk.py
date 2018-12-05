import numpy as _np
import pynumtools.kmc as _kmc
import itertools

import readdy_learn.analyze.basis as _basis


N_STIMULUS = 1
# ids             0    1         2          3        4         5       6        7     8
SPECIES_NAMES = ["S", "MAPKKK", "MAPKKK*", "MAPKK", "MAPKK*", "MAPK", "MAPK*", "TF", "TF*"]
N_SPECIES = len(SPECIES_NAMES)
TIMESTEP = 1e-3

INITIAL_STATES = [
    [N_STIMULUS, 1, 0, 1, 0, 1, 0, 1, 0]
]


ALL_BOGUS_CONVERSION_OPS = [
    lambda result: result.add_double_conversion((1, 3), (1, 4)),  # 9:  MAPKKK + MAPKK -> MAPKKK + MAPKK*
    lambda result: result.add_double_conversion((1, 5), (1, 6)),  # 10: MAPKKK + MAPK -> MAPK*
    lambda result: result.add_double_conversion((1, 7), (1, 8)),  # 11: MAPKKK + TF -> MAPKKK + TF*
    lambda result: result.add_double_conversion((2, 5), (2, 6)),  # 12: MAPKKK* + MAPK -> MAPKKK* + MAPK*
    lambda result: result.add_double_conversion((2, 7), (2, 8)),  # 13: MAPKKK* + TF -> MAPKKK* + TF*
    lambda result: result.add_double_conversion((3, 7), (3, 8)),  # 14: MAPKK + TF -> MAPKK + TF*
    lambda result: result.add_double_conversion((4, 7), (4, 8)),  # 15: MAPKK* + TF -> MAPKK* + TF*
    lambda result: result.add_double_conversion((5, 7), (5, 8)),  # 16: MAPK + TF -> MAPK + TF*                   wrong kinese partners
    lambda result: result.add_double_conversion((3, 5), (3, 6)),  # 17: MAPKK + MAPK -> MAPKK + MAPK*             wrong kinese partners
    lambda result: result.add_double_conversion((1, 4), (1, 3)),  # 18: MAPKKK + MAPKK* -> MAPKKK + MAPKK
    lambda result: result.add_double_conversion((1, 6), (1, 5)),  # 19: MAPKKK + MAPK* -> MAPKKK + MAPK
    lambda result: result.add_double_conversion((1, 8), (1, 7)),  # 20: MAPKKK + TF* -> MAPKKK + TF
    lambda result: result.add_double_conversion((2, 4), (2, 3)),  # 21: MAPKKK* + MAPKK* -> MAPKKK* + MAPKK       backward deactivation
    lambda result: result.add_double_conversion((2, 6), (2, 5)),  # 22: MAPKKK* + MAPK* -> MAPKKK* + MAPK         backward deactivation
    lambda result: result.add_double_conversion((2, 8), (2, 7)),  # 23: MAPKKK* + TF* -> MAPKKK* + TF             backward deactivation
    lambda result: result.add_double_conversion((3, 6), (3, 5)),  # 24: MAPKK + MAPK* -> MAPKK + MAPK
    lambda result: result.add_double_conversion((3, 8), (3, 7)),  # 25: MAPKK + TF* -> MAPKK + TF
    lambda result: result.add_double_conversion((4, 6), (4, 5)),  # 26: MAPKK* + MAPK* -> MAPKK* + MAPK           backward deactivation
    lambda result: result.add_double_conversion((4, 8), (4, 7)),  # 27: MAPKK* + TF* -> MAPKK* + TF               backward deactivation
    lambda result: result.add_double_conversion((5, 8), (5, 7)),  # 28: MAPK + TF* -> MAPK + TF
    lambda result: result.add_double_conversion((6, 8), (6, 7)),  # 29: MAPK* + TF* -> MAPK* + TF                 backward deactivation
]


def n_combinations():
    return len(list(itertools.combinations(_np.arange(len(ALL_BOGUS_CONVERSION_OPS)), 15)))


def conversion_ops_range(start, end):
    return list(itertools.combinations(_np.arange(len(ALL_BOGUS_CONVERSION_OPS)), 15))[start:end]


class MAPKConfiguration(object):

    def __init__(self, conversion_ops_selection=_np.arange(len(ALL_BOGUS_CONVERSION_OPS))):
        self.conversion_ops = [ALL_BOGUS_CONVERSION_OPS[i] for i in conversion_ops_selection]
        self.n_bogus = len(self.conversion_ops)
        self.n_real = 8
        self.n_total = self.n_real + self.n_bogus
        self.rates = _np.array([
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,
            1.,
        ])
        self.rates = _np.concatenate((self.rates, _np.zeros((self.n_bogus,))))

    def bfc(self) -> _basis.BasisFunctionConfiguration:
        result = _basis.BasisFunctionConfiguration(N_SPECIES)

        result.add_double_conversion([0, 1], [0, 2])  # 1: S + MAPKKK -> S + MAPKKK*
        result.add_conversion(2, 1)                   # 2: MAPKKK* -> MAPKKK
        result.add_double_conversion([2, 3], [2, 4])  # 3: MAPKKK* + MAPKK -> MAPKKK* + MAPKK*
        result.add_conversion(4, 3)                   # 4: MAPKK* -> MAPKK
        result.add_double_conversion([4, 5], [4, 6])  # 5: MAPKK* + MAPK -> MAPKK* + MAPK*
        result.add_conversion(6, 5)                   # 6: MAPK* -> MAPK
        result.add_double_conversion([6, 7], [6, 8])  # 7: MAPK* + TF -> MAPK* + TF*
        result.add_conversion(8, 7)                   # 8: TF* -> TF

        for op in self.conversion_ops:
            op(result)

        return result


def generate_kmc(target_time: float, n_realizations: int, config: MAPKConfiguration, njobs: int):
    import readdy_learn.analyze.generate as generate
    sys = _kmc.ReactionDiffusionSystem(diffusivity=len(SPECIES_NAMES) * [[[0.]]],
                                       n_species=len(SPECIES_NAMES), n_boxes=1,
                                       init_state=[INITIAL_STATES[0]],
                                       species_names=SPECIES_NAMES)
    sys.add_double_conversion(["S", "MAPKKK"], ["S", "MAPKKK*"], _np.array([config.rates[0]]))
    sys.add_conversion("MAPKKK*", "MAPKKK", _np.array([config.rates[1]]))
    sys.add_double_conversion(["MAPKKK*", "MAPKK"], ["MAPKKK*", "MAPKK*"], _np.array([config.rates[2]]))
    sys.add_conversion("MAPKK*", "MAPKK", _np.array([config.rates[3]]))
    sys.add_double_conversion(["MAPKK*", "MAPK"], ["MAPKK*", "MAPK*"], _np.array([config.rates[4]]))
    sys.add_conversion("MAPK*", "MAPK", _np.array([config.rates[5]]))
    sys.add_double_conversion(["MAPK*", "TF"], ["MAPK*", "TF*"], _np.array([config.rates[6]]))
    sys.add_conversion("TF*", "TF", _np.array([config.rates[7]]))

    _, counts = generate.generate_averaged_kmc_counts(lambda: sys, target_time, TIMESTEP,
                                                      n_realizations=n_realizations, njobs=njobs)
    times = _np.linspace(0, counts.shape[0] * TIMESTEP, endpoint=False, num=counts.shape[0])

    for sp in range(len(SPECIES_NAMES)):
        x = counts[:, sp]
        indices = 1 + _np.where(x[:-1] != x[1:])[0]
        indices = _np.insert(indices, 0, 0)
        interpolated = _np.interp(times, times[indices], x[indices])
        counts[:, sp] = interpolated

    dcounts_dt = derivative(times, counts)

    return times, counts, dcounts_dt


def derivative(times, counts):
    dcounts_dt = _np.empty_like(counts)
    for sp in range(N_SPECIES):
        x = counts[:, sp]

        indices = 1 + _np.where(x[:-1] != x[1:])[0]
        indices = _np.insert(indices, 0, 0)

        interpolated = _np.interp(times, times[indices], x[indices])
        interpolated = _np.gradient(interpolated) / TIMESTEP

        dcounts_dt[:, sp] = interpolated
    return dcounts_dt


def generate_lma(target_time: float, config: MAPKConfiguration):
    import readdy_learn.analyze.generate as generate
    basis = config.bfc()
    assert basis.n_basis_functions == len(config.rates)
    _, counts = generate.generate_continuous_counts(config.rates, INITIAL_STATES[0],
                                                    basis, TIMESTEP, target_time / TIMESTEP,
                                                    noise_variance=0, n_realizations=1)
    times = _np.linspace(0, counts.shape[0] * TIMESTEP, endpoint=False, num=counts.shape[0])

    for sp in range(len(SPECIES_NAMES)):
        x = counts[:, sp]
        indices = 1 + _np.where(x[:-1] != x[1:])[0]
        indices = _np.insert(indices, 0, 0)
        interpolated = _np.interp(times, times[indices], x[indices])
        counts[:, sp] = interpolated

    dcounts_dt = derivative(times, counts)

    return times, counts, dcounts_dt


def solve(config: MAPKConfiguration, counts, dcounts_dt, alpha, l1_ratio):
    import readdy_learn.analyze.estimator as rlas
    import readdy_learn.analyze.tools as tools

    tolerances_to_try = _np.logspace(-16, -10, num=7)

    if isinstance(counts, (list, tuple)):
        assert isinstance(dcounts_dt, (list, tuple))
        counts = _np.concatenate(counts, axis=0).squeeze()
        dcounts_dt = _np.concatenate(dcounts_dt, axis=0).squeeze()
    else:
        assert isinstance(counts, _np.ndarray) and isinstance(dcounts_dt, _np.ndarray)

    traj = tools.Trajectory(counts, time_step=TIMESTEP)
    traj.dcounts_dt = dcounts_dt

    estimator = None
    for tol in tolerances_to_try:
        print("Trying tolerance {}".format(tol))
        estimator = rlas.ReaDDyElasticNetEstimator([traj], config.bfc(), alpha=alpha, l1_ratio=l1_ratio,
                                                   maxiter=5000, method='SLSQP', verbose=True, approx_jac=False,
                                                   options={'ftol': tol}, rescale=False,
                                                   init_xi=_np.zeros_like(config.rates),
                                                   constrained=True)

        estimator.fit(None)
        if estimator.success_:
            return estimator.coefficients_
    if estimator is not None:
        raise ValueError('*_*: {}, {}'.format(estimator.result_.status, estimator.result_.message))
    else:
        raise ValueError('-_-')


def solve_grid(config: MAPKConfiguration, counts, dcounts_dt, alphas, l1_ratios, njobs=1):
    import itertools
    import pathos.multiprocessing as multiprocessing
    from readdy_learn.analyze.progress import Progress

    alphas = _np.atleast_1d(_np.array(alphas).squeeze())
    lambdas = _np.atleast_1d(_np.array(l1_ratios).squeeze())
    params = itertools.product(alphas, lambdas)
    params = [(counts, dcounts_dt, p[0], p[1]) for p in params]

    progress = Progress(len(params), label="validation", nstages=1)

    def worker(args):
        c, dc, a, l = args
        try:
            rates = solve(config, c, dc, a, l)
        except ValueError as e:
            print(f"caught value error {e}")
            rates = _np.array([_np.nan for _ in range(len(config.rates))])
        return a, l, rates

    result = []
    with multiprocessing.Pool(processes=njobs) as p:
        for idx, res in enumerate(p.imap_unordered(worker, params, 1)):
            result.append(res)
            progress.increase()
    progress.finish()

    return result


def cv(config: MAPKConfiguration, counts, dcounts_dt, alphas=(1.,), l1_ratios=(1.,), n_splits=5, njobs=1):
    import readdy_learn.analyze.tools as tools
    import readdy_learn.analyze.cross_validation as cross_validation

    traj = tools.Trajectory(counts, time_step=TIMESTEP)
    traj.dcounts_dt = dcounts_dt
    cv = cross_validation.CrossValidation([traj], config.bfc())
    cv.splitter = 'kfold'
    cv.n_splits = n_splits
    cv.njobs = njobs
    cv.show_progress = True
    cv_result = cv.cross_validate(alphas, l1_ratios, realizations=1)
    result = {"cv_result": cv_result}
    return result
