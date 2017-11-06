import unittest

import numpy as np

import readdy_learn.analyze.generate as generate
import readdy_learn.generate.generate_tools.kinetic_monte_carlo as kmc
import readdy_learn.analyze.basis as basis
import readdy_learn.analyze.estimator as rlas
import readdy_learn.analyze.tools as tools

import matplotlib.pyplot as plt


class TestDeriv(unittest.TestCase):

    def test_derivative(self):
        """
        test the jac implementation against an approximated one!
        """
        plot = True
        desired_rates = np.array([.02, .07])
        init_condition = [[70, 50]]
        n_steps = 5000
        dt = 5e-6

        bfc = basis.BasisFunctionConfiguration(n_species=2)
        bfc.add_conversion(0, 1)  # A -> B
        bfc.add_conversion(1, 0)  # B -> A
        times_ode, counts_ode = generate.generate_continuous_counts(desired_rates, init_condition, bfc, dt, n_steps)

        traj = tools.Trajectory(counts_ode, dt, interpolation_degree='pw_linear')
        traj.update()
        estimator = rlas.ReaDDyElasticNetEstimator([traj], bfc, alpha=0., init_xi=init_condition, verbose=True)
        jac = estimator.get_analytical_jac()
        jac_approx = estimator.get_approximated_jac()

        rates_ab = np.arange(.001, .03, step=.001)
        rates_ba = desired_rates[1] * np.ones_like(rates_ab)
        ys = np.empty(shape=(len(rates_ab), 2))
        ys_approx = np.empty_like(ys)
        for i in range(len(rates_ab)):
            ys[i] = jac(np.array([rates_ab[i], rates_ba[i]]))
            ys_approx[i] = jac_approx(np.array([rates_ab[i], rates_ba[i]]))
        if plot:
            plt.plot(rates_ab, ys[:, 0], label='analytical jac d(a->b)')
            plt.plot(rates_ab, ys_approx[:, 0], label='approximated jac d(a->b)')
            plt.plot(rates_ab, ys[:, 1], label='analytical jac d(b->a)')
            plt.plot(rates_ab, ys_approx[:, 1], label='approximated jac d(b->a)')
            plt.legend()
            plt.show()
        np.testing.assert_allclose(ys, ys_approx, rtol=1e-4, atol=1e-4)

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
