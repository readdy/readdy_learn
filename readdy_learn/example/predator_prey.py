import numpy as _np
import pynumtools.kmc as _kmc

import readdy_learn.analyze.basis as _basis

FRICTION_PREY = .1
FRICTION_PREDATOR = .1

BETA = 1

INITIAL_STATES = _np.array([
    [1, 1],
])

RATES = _np.array([
    .1,  # X + X -> 0
    .1,  # Y + Y -> 0
    1 * BETA,  # X -> X + X
    BETA,  # X + Y -> Y + Y
    1 * BETA,  # Y -> 0
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # bogus
])

SPECIES_NAMES = ["X", "Y"]
TIMESTEP = 2e-7


def bfc():
    result = _basis.BasisFunctionConfiguration(len(SPECIES_NAMES))

    result.add_fusion(0, 0, None)                 # 1:  X + X -> 0
    result.add_fusion(1, 1, None)                 # 2:  Y + Y -> 0
    result.add_fission(0, 0, 0)                   # 3:  X     -> X + X
    result.add_double_conversion([0, 1], [1, 1])  # 4:  X + Y -> Y + Y
    result.add_decay(1)                           # 5:  Y     -> 0

    result.add_double_conversion([0, 1], [0, 0])  # 6:  X + Y -> X + X
    result.add_decay(0)                           # 7:  X     -> 0
    result.add_fusion(1, 1, 1)                    # 8:  Y + Y -> Y
    result.add_fission(1, 1, 1)                   # 9:  Y     -> Y + Y
    result.add_fusion(0, 0, 0)                    # 10: X + X -> X
    result.add_fusion(0, 1, 0)                    # 11: X + Y -> X
    result.add_fusion(0, 1, 1)                    # 12: X + Y -> Y
    result.add_fission(0, 0, 1)                   # 13: X + X -> Y
    result.add_conversion(0, 1)                   # 14: X     -> Y
    result.add_conversion(1, 0)                   # 15: Y     -> X
    result.add_fission(0, 1, 1)                   # 16: X     -> Y + Y

    return result


def derivative(times, counts):
    dcounts_dt = _np.empty_like(counts)
    for sp in range(len(SPECIES_NAMES)):
        x = counts[:, sp]

        indices = 1 + _np.where(x[:-1] != x[1:])[0]
        indices = _np.insert(indices, 0, 0)

        interpolated = _np.interp(times, times[indices], x[indices])
        interpolated = _np.gradient(interpolated) / TIMESTEP

        dcounts_dt[:, sp] = interpolated
    return dcounts_dt


def generate_lma(initial_state: int, target_time: float):
    import readdy_learn.analyze.generate as generate
    _, counts = generate.generate_continuous_counts(RATES, INITIAL_STATES[initial_state],
                                                    bfc(), TIMESTEP, target_time / TIMESTEP,
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


def generate_kmc(initial_state: int, target_time: float, n_realizations: int, njobs: int):
    import readdy_learn.analyze.generate as generate
    sys = _kmc.ReactionDiffusionSystem(diffusivity=len(SPECIES_NAMES) * [[[0.]]],
                                       n_species=len(SPECIES_NAMES), n_boxes=1,
                                       init_state=[INITIAL_STATES[initial_state]],
                                       species_names=SPECIES_NAMES)
    sys.add_fusion("X", "X", None, _np.array([RATES[0]]))
    sys.add_fusion("Y", "Y", None, _np.array([RATES[1]]))
    sys.add_fission("X", "X", "X", _np.array([RATES[2]]))
    sys.add_double_conversion(["X", "Y"], ["Y", "Y"], _np.array([RATES[3]]))
    sys.add_decay("Y", _np.array([RATES[4]]))

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
                                                   constrained=True)

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

    alphas = _np.atleast_1d(_np.array(alphas).squeeze())
    lambdas = _np.atleast_1d(_np.array(l1_ratios).squeeze())
    params = itertools.product(alphas, lambdas)
    params = [(counts, dcounts_dt, p[0], p[1]) for p in params]

    def worker(args):
        c, dc, a, l = args
        return a, l, solve(c, dc, a, l)

    result = []
    with multiprocessing.Pool(processes=njobs) as p:
        for idx, res in enumerate(p.imap_unordered(worker, params, 1)):
            result.append(res)

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
    cv_result = cv.cross_validate(alphas, l1_ratios, realizations=1)
    result = {"cv_result": cv_result}
    return result
