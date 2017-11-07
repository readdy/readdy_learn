import numpy as _np
from scipy.integrate import odeint as _odeint
from pathos.multiprocessing import Pool as _Pool


def generate_continuous_counts(rates, initial_condition, bfc, timestep, n_steps):
    def fun_reference(data, _):
        theta = _np.array([f(data) for f in bfc.functions])
        return _np.matmul(rates, theta)

    xs = _np.linspace(0, n_steps*timestep, num=n_steps, endpoint=False)
    initial_condition = _np.array(initial_condition).squeeze()
    return xs, _odeint(fun_reference, initial_condition, xs)


def generate_kmc_counts(set_up_system, n_kmc_steps, timestep):
    sys = set_up_system()
    sys.simulate(n_kmc_steps)
    counts, times, _ = sys.get_counts_config(timestep=timestep)
    return times, counts


def generate_averaged_kmc_counts(set_up_system, n_kmc_steps, timestep, n_realizations, njobs=8):

    params = [(set_up_system, n_kmc_steps, timestep) for _ in range(n_realizations)]

    try:
        def generate_wrapper(args):
            _, counts = generate_kmc_counts(*args)
            return counts

        avgcounts = None
        N = 0.

        with _Pool(processes=njobs) as p:
            for counts in p.imap(generate_wrapper, params, 1):
                N += 1.
                if avgcounts is None:
                    avgcounts = counts
                else:
                    if counts.shape[0] < avgcounts.shape[0]:
                        avgcounts = avgcounts[:counts.shape[0],:]
                        avgcounts += counts
                    else:
                        avgcounts += counts[:avgcounts.shape[0],:]

        avgcounts /= N
        times = _np.linspace(0, float(avgcounts.shape[0]) * float(timestep), num=avgcounts.shape[0], endpoint=False)
        assert times.shape[0] == avgcounts.shape[0]
        if not _np.isclose(times[1] - times[0], timestep):
            print("WARN: times[1]-times[0] was {} but was expected to be {}".format(times[1]-times[0], timestep))
        return times, avgcounts
    finally:
        pass

def generate_kmc_events(set_up_system, n_kmc_steps):
    sys = set_up_system()
    sys.simulate(n_kmc_steps)
    counts, times, state = sys.sequence
    return times, _np.array(state).squeeze()
#
# def generate_averaged_kmc_events(set_up_system, n_kmc_steps, n_realizations):
#
#     times, counts = None, None
#     for _ in range(n_realizations):
#         ttimes, ccounts = generate_kmc_events(set_up_system, n_kmc_steps)
#         if times is None:
#             times, counts = ttimes, ccounts
#         else:
#             indices = _np.searchsorted(times, ttimes)