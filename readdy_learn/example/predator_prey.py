import numpy as _np
import pynumtools.kmc as _kmc

import readdy_learn.analyze.basis as _basis

RATES = _np.array([
    1.,  # X + X -> 0
    1.,  # Y + Y -> 0
    1.,  # X -> X + X
    1.,  # X + Y -> Y + Y
    1.,  # Y -> 0
])

SPECIES_NAMES = ["X", "Y"]
INITIAL_STATES = _np.array([
    [10, 3],
])
TIMESTEP = 1e-3


def bfc():
    result = _basis.BasisFunctionConfiguration(len(SPECIES_NAMES))

    result.add_fusion(0, 0, None)
    result.add_fusion(1, 1, None)
    result.add_fission(0, 0, 0)
    result.add_double_conversion([0, 1], [1, 1])
    result.add_decay(1)

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

    return counts, dcounts_dt


def generate_kmc(initial_state: int, target_time: float, n_realizations: int, njobs: int):
    import readdy_learn.analyze.generate as generate
    sys = _kmc.ReactionDiffusionSystem(diffusivity=len(SPECIES_NAMES) * [[[0.]]],
                                       n_species=len(SPECIES_NAMES), n_boxes=1,
                                       init_state=INITIAL_STATES[initial_state],
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

    return counts, dcounts_dt


def solve(counts, dcounts_dt, alpha, l1_ratio):
    import readdy_learn.analyze.estimator as rlas
    import readdy_learn.analyze.tools as tools

    tolerances_to_try = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8]

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
