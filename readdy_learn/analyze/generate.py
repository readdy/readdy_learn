import numpy as _np
from scipy.integrate import odeint as _odeint
from pathos.multiprocessing import Pool as _Pool


def generate_continuous_counts(rates, initial_condition, bfc, timestep, n_steps):
    def fun_reference(data, _):
        theta = _np.array([f(data) for f in bfc.functions])
        return _np.matmul(rates, theta)

    xs = _np.linspace(0, n_steps*timestep, num=timestep, endpoint=False)
    initial_condition = _np.array(initial_condition).squeeze()
    return xs, _odeint(fun_reference, initial_condition, xs)


def generate_kmc_counts(set_up_system, n_kmc_steps, timestep):
    sys = set_up_system()
    sys.simulate(n_kmc_steps)
    counts, times, _ = sys.get_counts_config(timestep=timestep)
    return times, counts


def generate_averaged_kmc_counts(set_up_system, n_kmc_steps, timestep, n_realizations, njobs=8):

    params = [(set_up_system, n_kmc_steps, timestep) for _ in range(n_realizations)]

    def generate_wrapper(args):
        _, counts = generate_kmc_counts(*args)
        return counts

    with _Pool(processes=njobs) as p:
        counts = p.map(generate_wrapper, params, 1)

    min_n_times = -1
    min_n_times_ix = 0
    for ix, cc in enumerate(counts):
        if min_n_times == -1:
            min_n_times = cc.shape[0]
            min_n_times_ix = ix
        else:
            if cc.shape[0] < min_n_times:
                min_n_times = cc.shape[0]
                min_n_times_ix = ix
    avgcounts = _np.empty_like(counts[min_n_times_ix])
    ncounts = avgcounts.shape[0]
    assert ncounts == min_n_times
    for i in range(len(counts)):
        counts[i] = counts[i][:ncounts, :]
    counts = _np.array(counts)
    avgcounts = _np.average(counts, axis=0)
    times = _np.linspace(0, float(avgcounts.shape[0]) * float(timestep), num=avgcounts.shape[0], endpoint=False)
    assert times.shape[0] == avgcounts.shape[0]
    if not _np.isclose(times[1] - times[0], timestep):
        print("WARN: times[1]-times[0] was {} but was expected to be {}".format(times[1]-times[0], timestep))
    return times, avgcounts
