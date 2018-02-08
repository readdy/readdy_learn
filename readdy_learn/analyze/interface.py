import typing as _typing
import numpy as _np


class ReactionAnalysisObject(object):

    @property
    def bfc(self):
        raise NotImplementedError("please implement me")

    @bfc.setter
    def bfc(self, value):
        raise NotImplementedError("please implement me")

    @property
    def recompute(self):
        raise NotImplementedError("please implement me")

    @recompute.setter
    def recompute(self, value):
        raise NotImplementedError("please implement me")

    @property
    def set_up_system(self):
        raise NotImplementedError("please implement me")

    @set_up_system.setter
    def set_up_system(self, value):
        raise NotImplementedError("please implement me")

    @property
    def recompute_traj(self):
        raise NotImplementedError("please implement me")

    @recompute_traj.setter
    def recompute_traj(self, value):
        raise NotImplementedError("please implement me")

    @property
    def fname_prefix(self):
        raise NotImplementedError("please implement me")

    @fname_prefix.setter
    def fname_prefix(self, value):
        raise NotImplementedError("please implement me")

    @property
    def fname_postfix(self):
        raise NotImplementedError("please implement me")

    @fname_postfix.setter
    def fname_postfix(self, value):
        raise NotImplementedError("please implement me")

    @property
    def n_species(self):
        raise NotImplementedError("please implement me")

    @n_species.setter
    def n_species(self, value):
        raise NotImplementedError("please implement me")

    @property
    def species_names(self):
        raise NotImplementedError("please implement me")

    @property
    def desired_rates(self):
        raise NotImplementedError("please implement me")

    @desired_rates.setter
    def desired_rates(self, value):
        raise NotImplementedError("please implement me")

    @property
    def best_alphas(self):
        raise NotImplementedError("please implement me")

    @property
    def initial_states(self):
        raise NotImplementedError("please implement me")

    @initial_states.setter
    def initial_states(self, value):
        raise NotImplementedError("please implement me")

    @property
    def trajs(self):
        raise NotImplementedError("please implement me")

    def generate_or_load_traj_lma(self, n, target_time, update_and_persist=False, noise_variance=0, realizations=1):
        raise NotImplementedError("please implement me")

    def solve(self, n, alpha, l1_ratio, tol=1e-12, constrained=True, recompute=False, verbose=True, persist=True):
        raise NotImplementedError("please implement me")

    def score(self, test_traj_n, rates):
        raise NotImplementedError("please implement me")


class ReactionLearnDataContainer(object):

    def __init__(self, counts: _np.ndarray):
        self._separate_derivs = {}
        self._counts = counts

    @property
    def separate_derivs(self) -> dict:
        return self._separate_derivs

    @property
    def counts(self) -> _np.ndarray:
        return self._counts

    @counts.setter
    def counts(self, value: _np.ndarray):
        self._counts = value

    @property
    def dcounts_dt(self) -> _typing.Optional[_np.ndarray]:
        if len(self.separate_derivs.keys()) != self.n_species:
            print("Dont have derivative (got {} but need {})".format(len(self.separate_derivs.keys()), self.n_species))
            return None
        deriv = _np.empty_like(self.counts)
        for s in range(self.n_species):
            deriv[:, s] = self.separate_derivs[s]
        return deriv

    @dcounts_dt.setter
    def dcounts_dt(self, value: _np.ndarray):
        for s in range(self.n_species):
            self.separate_derivs[s] = value[:, s]

    @property
    def n_species(self) -> int:
        return self.counts.shape[1]

    @property
    def n_time_steps(self):
        return self.counts.shape[0]


class AnalysisObjectGenerator(object):

    def generate_analysis_object(self, fname_prefix=None, fname_postfix=None) -> ReactionAnalysisObject:
        raise NotImplementedError("please implement me")

    @property
    def initial_states(self):
        raise NotImplementedError("please implement me")

    @initial_states.setter
    def initial_states(self, value):
        raise NotImplementedError("please implement me")

    @property
    def target_time(self):
        raise NotImplementedError("please implement me")

    @target_time.setter
    def target_time(self, value):
        raise NotImplementedError("please implement me")

    @property
    def noise_variance(self):
        raise NotImplementedError("please implement me")

    @noise_variance.setter
    def noise_variance(self, value):
        raise NotImplementedError("please implement me")

    def compute_gradient_derivatives(self, analysis: ReactionAnalysisObject, persist: bool = True):
        raise NotImplementedError("ok")

    @property
    def desired_rates(self):
        raise NotImplementedError("desired rates please")

    @desired_rates.setter
    def desired_rates(self, value):
        raise NotImplementedError("desired rates please")
