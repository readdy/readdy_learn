import progress_reporter as _pr


class Progress(_pr.ProgressReporter):
    def __init__(self, n=100, label=""):
        self._progress_register(amount_of_work=n, description=label)

    def increase(self, increment=1):
        self._progress_update(increment)

    def set_label(self, label=""):
        self._progress_set_description(0, label)

    def finish(self):
        self._progress_force_finish()
