from random import shuffle

import numpy as _np
import pynumtools.kmc as _kmc

import readdy_learn.analyze.basis as _basis

conversions_ops = [
    lambda result: result.add_double_conversion((1, 3), (1, 4)),
    lambda result: result.add_double_conversion((1, 5), (1, 6)),
    lambda result: result.add_double_conversion((1, 7), (1, 8)),
    lambda result: result.add_double_conversion((2, 5), (2, 6)),
    lambda result: result.add_double_conversion((2, 7), (2, 8)),
    lambda result: result.add_double_conversion((3, 7), (3, 8)),
    lambda result: result.add_double_conversion((4, 7), (4, 8)),
    # lambda result: result.add_double_conversion((5, 7), (5, 8)),  # wrong kinese partners
    lambda result: result.add_double_conversion((3, 5), (3, 6)),  # wrong kinese partners
    # lambda result: result.add_double_conversion((1, 4), (1, 3)),
    # lambda result: result.add_double_conversion((1, 6), (1, 5)),
    # lambda result: result.add_double_conversion((1, 8), (1, 7)),
    # lambda result: result.add_double_conversion((2, 4), (2, 3)),  # backward deactivation
    # lambda result: result.add_double_conversion((2, 6), (2, 5)),  # backward deactivation
    # lambda result: result.add_double_conversion((2, 8), (2, 7)),  # backward deactivation
    # lambda result: result.add_double_conversion((3, 6), (3, 5)),
    # lambda result: result.add_double_conversion((3, 8), (3, 7)),
    # lambda result: result.add_double_conversion((4, 6), (4, 5)),  # backward deactivation
    # lambda result: result.add_double_conversion((4, 8), (4, 7)),  # backward deactivation
    # lambda result: result.add_double_conversion((5, 8), (5, 7)),
    # lambda result: result.add_double_conversion((6, 8), (6, 7)),  # backward deactivation
]

N_BOGUS = len(conversions_ops)


def bogus_bfc(result: _basis.BasisFunctionConfiguration):
    for i in range(_np.min((N_BOGUS, len(conversions_ops)))):
        conversions_ops[i](result)
    return N_BOGUS


def bfc():
    result = _basis.BasisFunctionConfiguration(N_SPECIES)

    result.add_double_conversion([0, 1], [0, 2])  # S + MAPKKK -> S + MAPKKK*
    result.add_conversion(2, 1)  # MAPKKK* -> MAPKKK
    result.add_double_conversion([2, 3], [2, 4])  # MAPKKK* + MAPKK -> MAPKKK* -> MAPKK*
    result.add_conversion(4, 3)  # MAPKK* -> MAPKK
    result.add_double_conversion([4, 5], [4, 6])  # MAPKK* + MAPK -> MAPKK* -> MAPK*
    result.add_conversion(6, 5)  # MAPK* -> MAPK
    result.add_double_conversion([6, 7], [6, 8])  # MAPK* + TF -> MAPK* + TF*
    result.add_conversion(8, 7)  # TF* -> TF

    bogus_bfc(result)

    return result


N_STIMULUS = 100
SPECIES_NAMES = ["S", "MAPKKK", "MAPKKK*", "MAPKK", "MAPKK*", "MAPK", "MAPK*", "TF", "TF*"]
N_SPECIES = len(SPECIES_NAMES)
TIMESTEP = 1e-3
RATES = _np.array([
    1.,
    1000.,
    1.,
    1000.,
    1.,
    1000.,
    1.,
    1000.,
])
RATES = _np.concatenate((RATES, _np.zeros((N_BOGUS,))))
INITIAL_STATES = [
    [N_STIMULUS, 1000, 0, 1000, 0, 1000, 0, 1000, 0]
]


def generate_kmc(initial_state: int, target_time: float, n_realizations: int, njobs: int):
    import readdy_learn.analyze.generate as generate
    sys = _kmc.ReactionDiffusionSystem(diffusivity=len(SPECIES_NAMES) * [[[0.]]],
                                       n_species=len(SPECIES_NAMES), n_boxes=1,
                                       init_state=[INITIAL_STATES[initial_state]],
                                       species_names=SPECIES_NAMES)
    sys.add_double_conversion(["S", "MAPKKK"], ["S", "MAPKKK*"], _np.array([RATES[0]]))
    sys.add_conversion("MAPKKK*", "MAPKKK", _np.array([RATES[1]]))
    sys.add_double_conversion(["MAPKKK*", "MAPKK"], ["MAPKKK*", "MAPKK*"], _np.array([RATES[2]]))
    sys.add_conversion("MAPKK*", "MAPKK", _np.array([RATES[3]]))
    sys.add_double_conversion(["MAPKK*", "MAPK"], ["MAPKK*", "MAPK*"], _np.array([RATES[4]]))
    sys.add_conversion("MAPK*", "MAPK", _np.array([RATES[5]]))
    sys.add_double_conversion(["MAPK*", "TF"], ["MAPK*", "TF*"], _np.array([RATES[6]]))
    sys.add_conversion("TF*", "TF", _np.array([RATES[7]]))

    assert initial_state < len(INITIAL_STATES)
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


def generate_lma(initial_state: int, target_time: float):
    import readdy_learn.analyze.generate as generate
    basis = bfc()
    assert basis.n_basis_functions == len(RATES)
    _, counts = generate.generate_continuous_counts(RATES, INITIAL_STATES[initial_state],
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


def solve(counts, dcounts_dt, alpha, l1_ratio):
    import readdy_learn.analyze.estimator as rlas
    import readdy_learn.analyze.tools as tools

    tolerances_to_try = _np.logspace(-16, -1, num=16)

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
        estimator = rlas.ReaDDyElasticNetEstimator([traj], bfc(), alpha=alpha, l1_ratio=l1_ratio,
                                                   maxiter=30000, method='SLSQP', verbose=True, approx_jac=False,
                                                   options={'ftol': tol}, rescale=False,
                                                   init_xi=_np.zeros_like(RATES),
                                                   constrained=False)

        estimator.fit(None)
        if estimator.success_:
            return estimator.coefficients_
    if estimator is not None:
        raise ValueError('*_*: {}, {}'.format(estimator.result_.status, estimator.result_.message))
    else:
        raise ValueError('-_-')


def solve_grid(counts, dcounts_dt, alphas, l1_ratios, njobs=1):
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
        return a, l, solve(c, dc, a, l)

    result = []
    with multiprocessing.Pool(processes=njobs) as p:
        for idx, res in enumerate(p.imap_unordered(worker, params, 1)):
            result.append(res)
            progress.increase()
    progress.finish()

    return result


def cv(counts, dcounts_dt, alphas=(1.,), l1_ratios=(1.,), n_splits=5, njobs=1):
    import readdy_learn.analyze.tools as tools
    import readdy_learn.analyze.cross_validation as cross_validation

    traj = tools.Trajectory(counts, time_step=TIMESTEP)
    traj.dcounts_dt = dcounts_dt
    cv = cross_validation.CrossValidation([traj], bfc())
    cv.splitter = 'kfold'
    cv.n_splits = n_splits
    cv.njobs = njobs
    cv.show_progress = True
    cv_result = cv.cross_validate(alphas, l1_ratios, realizations=1)
    result = {"cv_result": cv_result}
    return result
