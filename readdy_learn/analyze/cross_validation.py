import time as _time

import numpy as _np
import itertools as _itertools

import readdy_learn.analyze.interface as _interface
import readdy_learn.analyze.progress as _progress

import pathos.multiprocessing as _multiprocessing

class CrossValidation(object):

    def __init__(self, regulation_network: _interface.AnalysisObjectGenerator, show_progress: bool = True,
                 njobs: int = 8):
        self._regulation_network = regulation_network
        self._show_progress = show_progress
        self._njobs = njobs
        self.result_ = None

    def _cross_validate(self, args):
        alpha, l1_ratio, cutoff, i = args

        hash = (i + int(_time.time())) % (2 ** 32 - 1)
        _np.random.seed(hash)

        regulation_network = self.regulation_network
        analysis_train = self.regulation_network.generate_analysis_object(None, None)
        for i in range(len(regulation_network.initial_states)):
            analysis_train.generate_or_load_traj_lma(i, regulation_network.target_time,
                                                     noise_variance=regulation_network.noise_variance,
                                                     realizations=1)
        regulation_network.compute_gradient_derivatives(analysis_train, persist=False)



    def cross_validate(self, alphas, lambdas, cutoffs=None, realizations: int = 10):
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