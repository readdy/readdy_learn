import time as _time
import typing as _typing

import numpy as _np
import itertools as _itertools

import readdy_learn.analyze.interface as _interface
import readdy_learn.analyze.progress as _progress
import readdy_learn.analyze.estimator as _estimator

import pathos.multiprocessing as _multiprocessing

from sklearn.model_selection import KFold as _KFold

_TrajList = _typing.List[_interface.ReactionLearnDataContainer]


class CrossValidation(object):

    def __init__(self, trajs: _TrajList, bfc, _n_splits: int = 10, show_progress: bool = True, njobs: int = 8):
        self._show_progress = show_progress
        self._njobs = njobs
        self.result_ = None

        self._counts = _np.stack([t.counts for t in trajs], axis=0)
        self._dcounts_dt = _np.stack([t.dcounts_dt for t in trajs], axis=0)
        self._n_splits = _n_splits
        self._bfc = bfc

    def _obtain_trajs_subset(self, indices: _np.ndarray):
        counts = self._counts[indices]
        dcounts_dt = self._dcounts_dt[indices]

        container = _interface.ReactionLearnDataContainer(counts)
        container.dcounts_dt = dcounts_dt
        return container

    def _solve(self, traj, alpha, l1_ratio, tol=1e-16):
        initial_value = _np.zeros((self._bfc.n_basis_functions,), dtype=_np.float64)
        estimator = _estimator.ReaDDyElasticNetEstimator(traj, self._bfc, alpha=alpha, l1_ratio=l1_ratio,
                                                         maxiter=30000, method='SLSQP', verbose=False, approx_jac=False,
                                                         options={'ftol': tol}, rescale=False,
                                                         init_xi=initial_value,
                                                         constrained=True)

        estimator.fit(None)
        if estimator.success_:
            rates = estimator.coefficients_
            return rates
        else:
            raise ValueError('*_*: {}, {}'.format(estimator.result_.status, estimator.result_.message))

    def _score(self, traj, rates):
        initial_value = _np.zeros((self._bfc.n_basis_functions,), dtype=_np.float64)
        estimator = _estimator.ReaDDyElasticNetEstimator(traj, self._bfc, alpha=0, l1_ratio=0,
                                                         maxiter=30000, method='SLSQP', verbose=False, approx_jac=False,
                                                         options={'ftol': 1e-12}, rescale=False,
                                                         init_xi=initial_value,
                                                         constrained=True)
        estimator.coefficients_ = rates
        return estimator.score(range(0, traj.n_time_steps), traj.dcounts_dt)

    def _cross_validate(self, args):
        alpha, l1_ratio, cutoff, i = args

        hash = (i + int(_time.time())) % (2 ** 32 - 1)
        _np.random.seed(hash)

        n_steps_total = self._counts.shape[0]
        indices = _np.arange(n_steps_total)

        splitter = _KFold(n_splits=self.n_splits)
        scores = []
        for train, test in splitter.split(indices):
            train_traj = self._obtain_trajs_subset(train)
            test_traj = self._obtain_trajs_subset(test)
            tolerances_to_try = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8]

            rates = None
            for tol in tolerances_to_try:
                try:
                    rates = self._solve(train_traj, alpha, l1_ratio, tol=tol)
                    break
                except ValueError:
                    if tol == tolerances_to_try[len(tolerances_to_try) - 1]:
                        raise

            score = self._score(test_traj, rates)
            scores.append(score)
        return {'alpha': alpha, 'l1_ratio': l1_ratio, 'cutoff': cutoff, 'score': scores}

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
            for idx, res in enumerate(p.imap_unordered(self._cross_validate, params, 1)):
                result.append(res)
                if self.show_progress:
                    progress.increase()

        if self.show_progress:
            progress.finish()

        self.result_ = result
        return result

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

    @property
    def n_splits(self):
        return self._n_splits

    @n_splits.setter
    def n_splits(self, value: int):
        self._n_splits = value
