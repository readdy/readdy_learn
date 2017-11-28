import progress_reporter as _pr
import numpy as _np


class Progress(_pr.ProgressReporter):
    def __init__(self, n=100, nstages=1, label=""):
        if not isinstance(n, (list, tuple, _np.ndarray)):
            n = [n] * nstages
        assert len(n) == nstages, "number of stages must be equal to length of n, or n must not be list/tuple/array"
        for i in range(nstages):
            self._progress_register(amount_of_work=n[i], description=label + " {}".format(i), stage=i)

    def increase(self, increment=1, stage=0):
        self._progress_update(increment, stage=stage)

    def set_label(self, label=""):
        self._progress_set_description(0, label)

    def finish(self, stage=0):
        self._progress_force_finish(stage=stage)
