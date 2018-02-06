import readdy_learn.analyze.analyze as _ana


class AnalysisObjectGenerator(object):

    def generate_analysis_object(self, fname_prefix=None, fname_postfix=None) -> _ana.ReactionAnalysis:
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

    def compute_gradient_derivatives(self, analysis: _ana.ReactionAnalysis, persist: bool = True):
        raise NotImplementedError("ok")

    @property
    def desired_rates(self):
        raise NotImplementedError("desired rates please")

    @desired_rates.setter
    def desired_rates(self, value):
        raise NotImplementedError("desired rates please")
