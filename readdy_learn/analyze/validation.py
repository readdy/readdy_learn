import time as _time

import numpy as _np
import itertools as _itertools

import readdy_learn.analyze.interface as _interface
import readdy_learn.analyze.progress as _progress

import pathos.multiprocessing as _multiprocessing


class Validation(object):

    def __init__(self, regulation_network: _interface.AnalysisObjectGenerator, show_progress: bool = True,
                 njobs: int = 8):
        self._regulation_network = regulation_network
        self._show_progress = show_progress
        self._njobs = njobs
        self.result_ = None

    def _validate(self, args):
        alpha, l1_ratio, cutoff, i = args
        _np.random.seed(i + int(_time.time()))
        regulation_network = self.regulation_network
        analysis_train = self.regulation_network.generate_analysis_object(None, None)
        for i in range(len(regulation_network.initial_states)):
            analysis_train.generate_or_load_traj_lma(i, regulation_network.target_time,
                                                     noise_variance=regulation_network.noise_variance,
                                                     realizations=1)
        regulation_network.compute_gradient_derivatives(analysis_train, persist=False)

        analysis_test = self.regulation_network.generate_analysis_object(None, None)
        for i in range(len(regulation_network.initial_states)):
            analysis_test.generate_or_load_traj_lma(i, regulation_network.target_time,
                                                    noise_variance=regulation_network.noise_variance,
                                                    realizations=1)
        regulation_network.compute_gradient_derivatives(analysis_test, persist=False)

        tolerances_to_try = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8]

        rates = None
        for tol in tolerances_to_try:
            try:
                rates = analysis_train.solve(0, alpha, l1_ratio, tol=tol, constrained=True,
                                             recompute=True, verbose=False, persist=False)
                break
            except ValueError:
                if tol == 1e-12:
                    raise
        score = analysis_test.score(0, rates)
        return {'alpha': alpha, 'l1_ratio': l1_ratio, 'cutoff': cutoff, 'score': score}

    def validate(self, alphas, lambdas, cutoffs=None, realizations: int = 10):
        if cutoffs is None:
            cutoffs = _np.zeros((1,), dtype=float)
        else:
            cutoffs = _np.array(cutoffs).squeeze()
        alphas = _np.array(alphas).squeeze()
        lambdas = _np.array(lambdas).squeeze()

        assert _np.alltrue(lambdas <= 1), "some lambdas were greater than 1"
        assert _np.alltrue(lambdas >= 0), "some lambdas were smaller than 0"

        print("validating across grid with {} alphas, {} lambdas, {} cutoffs with {} realizations"
              .format(len(alphas), len(lambdas), len(cutoffs), realizations))

        params = _itertools.product(alphas, lambdas, cutoffs)
        params = [(p[0], p[1], p[2], j + i * realizations) for i, p in enumerate(params)
                  for j, _ in enumerate(range(realizations))]

        result = []

        progress = None
        if self.show_progress:
            progress = _progress.Progress(len(params),
                                          label="validation", nstages=1)

        with _multiprocessing.Pool(processes=self.njobs) as p:
            for idx, res in enumerate(p.imap_unordered(self._validate, params, 1)):
                result.append(res)
                if self.show_progress:
                    progress.increase()

        if self.show_progress:
            progress.finish()

        self.result_ = result
        return result

    @property
    def regulation_network(self):
        return self._regulation_network

    @regulation_network.setter
    def regulation_network(self, value: _interface.AnalysisObjectGenerator):
        self._regulation_network = value

    @property
    def show_progress(self):
        return self._show_progress

    @show_progress.setter
    def show_progress(self, value: bool):
        self._show_progress = value

    @property
    def njobs(self):
        return self._njobs

    @njobs.setter
    def njobs(self, value: int):
        self._njobs = value


def get_n_realizations(result):
    some_alpha = result[0]['alpha']
    some_l1_ratio = result[0]['l1_ratio']
    some_cutoff = result[0]['cutoff']
    count = 0
    for r in result:
        if r['alpha'] == some_alpha and r['l1_ratio'] == some_l1_ratio and r['cutoff'] == some_cutoff:
            count += 1
    return count


def get_distinct_l1_ratios(result):
    s = set()
    for r in result:
        s.add(r['l1_ratio'])
    return sorted(list(s))


def get_distinct_alphas(result):
    s = set()
    for r in result:
        s.add(r['alpha'])
    return sorted(list(s))


def get_scores(result, alpha, l1_ratio):
    return [r['score'] for r in result if r['alpha'] == alpha and r['l1_ratio'] == l1_ratio]


def plot_validation_result(result):
    import matplotlib.pyplot as plt
    l1_ratios = get_distinct_l1_ratios(result)
    alphas = get_distinct_alphas(result)
    for l1_ratio in l1_ratios:
        xs = _np.array(alphas)
        ys = _np.empty_like(xs)
        yerr = _np.empty_like(xs)
        for ix, alpha in enumerate(alphas):
            ys[ix] = _np.mean(get_scores(result, alpha, l1_ratio))
            yerr[ix] = _np.std(get_scores(result, alpha, l1_ratio))
        plt.errorbar(xs, ys, yerr=yerr)
