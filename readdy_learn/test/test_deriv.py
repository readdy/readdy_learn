import unittest

import numpy as np

import readdy_learn.analyze.generate as generate
import readdy_learn.generate.generate_tools.kinetic_monte_carlo as kmc
import readdy_learn.analyze.basis as basis

import matplotlib.pyplot as plt


class TestDeriv(unittest.TestCase):

    def test_generators(self):
        desired_rates = np.array([.02, .02, 1e-2, 1e-2])
        init_condition = [[70, 50, 30, 80]]

        def set_up_system():
            sys = kmc.ReactionDiffusionSystem(n_species=4, n_boxes=1, diffusivity=[[[0.]], [[0.]], [[0.]], [[0.]]],
                                              init_state=init_condition, species_names=["A", "B", "C", "D"])
            sys.add_conversion("A", "D", np.array([desired_rates[0]]))
            sys.add_conversion("D", "A", np.array([desired_rates[1]]))
            sys.add_fusion("A", "B", "C", np.array([desired_rates[2]]))
            sys.add_fission("C", "D", "B", np.array([desired_rates[3]]))
            return sys

        n_steps = 50
        dt = 5e-6

        times_akmc, counts_akmc = generate.generate_averaged_kmc_counts(set_up_system, n_steps, dt, 8)

        times_kmc, counts_kmc = generate.generate_kmc_counts(set_up_system, n_steps, dt)

        bfc = basis.BasisFunctionConfiguration(n_species=4)
        bfc.add_conversion(0, 3)  # A -> D
        bfc.add_conversion(3, 0)  # D -> A
        bfc.add_fusion(0, 1, 2)  # A + B -> C
        bfc.add_fission(2, 3, 1)  # C -> D + B
        times_ode, counts_ode = generate.generate_continuous_counts(desired_rates, init_condition, bfc, dt,
                                                                    max(times_kmc.shape[0], times_akmc.shape[0]))

        plt.plot(times_akmc, counts_akmc, label='kmc, averaged over 10')
        plt.plot(times_kmc, counts_kmc, label='kmc')
        plt.plot(times_ode, counts_ode, 'k--', label='ode')

        plt.show()


if __name__ == '__main__':
    pass
