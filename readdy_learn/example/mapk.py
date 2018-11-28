import numpy as _np
import pynumtools.kmc as _kmc

import readdy_learn.analyze.basis as _basis

SPECIES_NAMES = ["MAPKKK", "MPAKKK*", "MAPKK", "MPAKK*", "MAPK", "MPAK*", "TF", "TF*"]
N_SPECIES = len(SPECIES_NAMES)
TIMESTEP = 1e-3
RATES = _np.array([
    .1,
    .1,
    .1,
    .1
])
INITIAL_STATES = [
    [1000, 0, 1000, 0, 1000, 0, 1000, 0]
]


def bfc():
    result = _basis.BasisFunctionConfiguration(N_SPECIES)

    result.add_conversion(0, 1)
    result.add_double_conversion([1, 2], [1, 3])
    result.add_double_conversion([3, 4], [3, 5])
    result.add_double_conversion([5, 6], [5, 7])

    return result


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


