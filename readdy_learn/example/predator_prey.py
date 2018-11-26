import numpy as _np
import pynumtools.kmc as _kmc
import readdy_learn.analyze.basis as _basis
import matplotlib.pyplot as plt
from pathos.multiprocessing import Pool as _Pool
import readdy_learn.analyze.progress as _pr
import readdy_learn.analyze.interface as _interface
import readdy_learn.analyze.analyze as _ana
from pynumtools.finite_differences import fd_coefficients as _fd_coeffs
from pynumtools.util import sliding_window as _sliding_window

from readdy_learn.analyze.interface import ReactionAnalysisObject

DEFAULT_DESIRED_RATES = _np.array([
    1.,  # X + X -> 0
    1.,  # Y + Y -> 0
    1.,  # X -> X + X
    1.,  # X + Y -> Y + Y
    1.,  # Y -> 0
])


class PredatorPrey(_interface.AnalysisObjectGenerator):

    def __init__(self):
        self.n_species = 2
        self.species_names = ["X", "Y"]
        initial_states = [
            [10, 3],
        ]
        self._initial_states = [_np.array([arr]) for arr in initial_states]
        self.timestep = 1e-3
        self.ld_derivative_config = {
            'ld_derivative_atol': 1e-4,
            'ld_derivative_rtol': None,
            'ld_derivative_linalg_solver_maxit': 100000,
            'ld_derivative_alpha': 1e-1,
            'ld_derivative_solver': 'spsolve',
            'ld_derivative_linalg_solver_tol': 1e-10,
            'ld_derivative_use_preconditioner': False
        }
        self._target_time = 3.
        self._noise_variance = 1e-2

    def set_up_system(self, init_state):
        sys = _kmc.ReactionDiffusionSystem(diffusivity=self.n_species * [[[0.]]], n_species=self.n_species, n_boxes=1,
                                           init_state=init_state, species_names=self.species_names)
        # usual stuff A
        sys.add_fusion("X", "X", None, _np.array([self.desired_rates[0]]))
        sys.add_fusion("Y", "Y", None, _np.array([self.desired_rates[1]]))
        sys.add_fission("X", "X", "X", _np.array([self.desired_rates[2]]))
        sys.add_double_conversion(["X", "Y"], ["Y", "Y"], _np.array([self.desired_rates[3]]))
        sys.add_decay("Y", _np.array([self.desired_rates[4]]))
        return sys

    def generate_analysis_object(self, fname_prefix=None, fname_postfix=None) -> ReactionAnalysisObject:
        if fname_prefix is None:
            fname_prefix = "regulation_network"
        if fname_postfix is None:
            fname_postfix = ""
        analysis = _ana.ReactionAnalysis(self.get_bfc(), self.desired_rates, self.initial_states, self.set_up_system,
                                         fname_prefix=fname_prefix, fname_postfix=fname_postfix,
                                         n_species=self.n_species, timestep=self.timestep,
                                         ld_derivative_config=self.ld_derivative_config, recompute_traj=False,
                                         species_names=self.species_names)
        return analysis

    @property
    def initial_states(self):
        return self._initial_states

    @property
    def target_time(self):
        return self._target_time

    @target_time.setter
    def target_time(self, value):
        self._target_time = value

    @property
    def noise_variance(self):
        return self._noise_variance

    @noise_variance.setter
    def noise_variance(self, value):
        self._noise_variance = value

    def compute_gradient_derivatives(self, analysis: ReactionAnalysisObject, persist: bool = True, wwidth=None):
        for t in range(len(self.initial_states)):
            traj = analysis.get_traj(t)
            # for sp in [0, 3, 6]:
            #    dx = _np.zeros_like(traj.counts[:, sp])
            #    print("species {} dx.shape {}".format(sp, dx.shape))
            #    traj.separate_derivs[sp] = dx
            times = traj.times

            for sp in [0, 1]:
                x = traj.counts[:, sp]
                dt = traj.time_step

                indices = 1 + _np.where(x[:-1] != x[1:])[0]
                indices = _np.insert(indices, 0, 0)

                if wwidth is not None:
                    if len(_np.atleast_1d(indices.squeeze())) == 1:
                        indices = _np.arange(2 * wwidth + 1, dtype=int)

                    unique_times, unique_counts = times[indices].squeeze(), x[indices].squeeze()
                    unique_deriv = _np.empty_like(unique_counts)

                    for ix, (wx, wy) in enumerate(
                            zip(_sliding_window(_np.atleast_1d(unique_times), width=wwidth, fixed_width=False),
                                _sliding_window(_np.atleast_1d(unique_counts), width=wwidth, fixed_width=False))):
                        x = unique_times[ix]
                        coeff = _fd_coeffs(x, wx, k=1)
                        unique_deriv[ix] = coeff.dot(wy)

                    interpolated = _np.interp(times, unique_times, unique_deriv)
                    traj.separate_derivs[sp] = interpolated
                else:
                    interpolated = _np.interp(times, times[indices], x[indices])
                    interpolated = _np.gradient(interpolated) / dt

                    traj.separate_derivs[sp] = interpolated

            if persist:
                traj.persist()

    def interpolate_counts(self, analysis: _ana.ReactionAnalysis, persist: bool = True):
        for t in range(len(self.initial_states)):
            traj = analysis.get_traj(t)
            times = traj.times

            for sp in [0, 1]:
                x = traj.counts[:, sp]

                indices = 1 + _np.where(x[:-1] != x[1:])[0]
                indices = _np.insert(indices, 0, 0)

                interpolated = _np.interp(times, times[indices], x[indices])

                traj.counts[:, sp] = interpolated

            if persist:
                traj.persist()

    @property
    def desired_rates(self):
        return DEFAULT_DESIRED_RATES

    def get_bfc(self):
        bfc = _basis.BasisFunctionConfiguration(self.n_species)

        bfc.add_fusion(0, 0, None)
        bfc.add_fusion(1, 1, None)
        bfc.add_fission(0, 0, 0)
        bfc.add_double_conversion([0, 1], [1, 1])
        bfc.add_decay(1)

        return bfc
