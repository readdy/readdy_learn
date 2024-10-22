import numpy as _np
from scipy.integrate import odeint as _odeint
from pathos.multiprocessing import Pool as _Pool


def generate_continuous_counts(rates, initial_condition, bfc, timestep, n_steps, noise_variance=0., n_realizations=1,
                               njobs=8, supersample=1000):
    if n_realizations == 1 or noise_variance == 0.:
        def fun_reference(data, _):
            theta = _np.array([f(data) for f in bfc.functions])
            return _np.matmul(rates, theta)

        xs = _np.linspace(0, n_steps * timestep, num=int(n_steps*supersample), endpoint=False)
        initial_condition = _np.array(initial_condition).squeeze()
        ys = _np.array(_odeint(fun_reference, initial_condition, xs)).squeeze()
        if noise_variance > 0.:
            ys += _np.random.normal(0.0, _np.sqrt(noise_variance), size=ys.shape)
        return xs[::supersample], ys[::supersample]
    else:
        def fun_reference(data, _):
            theta = _np.array([f(data) for f in bfc.functions])
            return _np.matmul(rates, theta)

        xs = _np.linspace(0, n_steps * timestep, num=int(n_steps*supersample), endpoint=False)
        initial_condition = _np.array(initial_condition).squeeze()

        def generate_wrapper(args):
            ys = _np.array(_odeint(fun_reference, initial_condition, xs)).squeeze()
            if noise_variance > 0.:
                _np.random.seed(*args)
                ys += _np.random.normal(0.0, _np.sqrt(noise_variance), size=ys.shape)
            return ys

        avgcounts = None
        N = 0.

        params = [(i,) for i in range(n_realizations)]
        with _Pool(processes=njobs) as p:
            for counts in p.imap(generate_wrapper, params, 1):
                N += 1.
                if avgcounts is None:
                    avgcounts = counts
                else:
                    avgcounts += counts
            avgcounts /= N
        return xs[::supersample], avgcounts[::supersample]


def generate_kmc_counts(set_up_system, target_time, timestep):
    sys = set_up_system()
    sys.simulate(target_time=target_time)
    counts, times, _ = sys.get_counts_config(timestep=timestep)
    return times, counts


def generate_averaged_kmc_counts(set_up_system, target_time, timestep, n_realizations, njobs=8):
    import logging
    import pynumtools.kmc as kmc

    prevlevel = kmc.log.getEffectiveLevel()
    kmc.log.setLevel(logging.WARNING)
    params = [(set_up_system, target_time, timestep) for _ in range(n_realizations)]
    try:
        def generate_wrapper(args):
            _, counts = generate_kmc_counts(*args)
            return counts

        avgcounts = None
        N = 0.

        if njobs == 1:
            for p in params:
                counts = generate_wrapper(p)
                N += 1.
                if avgcounts is None:
                    avgcounts = counts
                else:
                    if counts.shape[0] < avgcounts.shape[0]:
                        avgcounts = avgcounts[:counts.shape[0], :]
                        avgcounts += counts
                    else:
                        avgcounts += counts[:avgcounts.shape[0], :]

        else:
            with _Pool(processes=njobs) as p:
                for counts in p.imap(generate_wrapper, params, 1):
                    N += 1.
                    if avgcounts is None:
                        avgcounts = counts
                    else:
                        if counts.shape[0] < avgcounts.shape[0]:
                            avgcounts = avgcounts[:counts.shape[0], :]
                            avgcounts += counts
                        else:
                            avgcounts += counts[:avgcounts.shape[0], :]

        avgcounts /= N
        times = _np.linspace(0, float(avgcounts.shape[0]) * float(timestep), num=avgcounts.shape[0], endpoint=False)
        assert times.shape[0] == avgcounts.shape[0]
        if not _np.isclose(times[1] - times[0], timestep):
            print("WARN: times[1]-times[0] was {} but was expected to be {}".format(times[1] - times[0], timestep))
        return times, avgcounts
    finally:
        kmc.log.setLevel(prevlevel)


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
