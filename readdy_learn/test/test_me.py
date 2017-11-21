import os
import numpy as np
import matplotlib as mpl

import readdy_learn.analyze.analyze as ana
import readdy_learn.analyze.basis as basis
import readdy_learn.generate.generate_tools.kinetic_monte_carlo as kmc
import matplotlib.pyplot as plt

desired_rates = np.array([2e-2, 2e-2, 1e-2, 1e-4, 1e-4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
initial_states = [
    [52, 42, 81, 7], [30, 5, 16, 47], [50, 5, 10, 10], [16, 53, 7, 40],
    [12, 11, 7, 87], [30, 5, 16, 47], [5, 12, 8, 72], [52, 42, 81, 7],
    [55, 32, 79, 24], [48, 39, 29, 5], [8, 83, 77, 32], [16, 55, 94, 78],
    [48, 5, 44, 10]
]
initial_states = [np.array([arr]) for arr in initial_states]
n_species = 4


def set_up_system(init_state):
    sys = kmc.ReactionDiffusionSystem(diffusivity=[[[0.]], [[0.]], [[0.]], [[0.]]], n_species=n_species, n_boxes=1,
                                      init_state=init_state, species_names=["A", "B", "C", "D"])
    sys.add_conversion("A", "D", np.array([desired_rates[0]]))
    sys.add_conversion("D", "A", np.array([desired_rates[1]]))
    sys.add_conversion("D", "B", np.array([desired_rates[2]]))
    sys.add_fusion("A", "B", "C", np.array([desired_rates[3]]))
    sys.add_fission("C", "D", "B", np.array([desired_rates[4]]))

    return sys


def get_bfc():
    bfc = basis.BasisFunctionConfiguration(n_species)
    bfc.add_conversion(0, 3)  # A -> D
    bfc.add_conversion(3, 0)  # D -> A
    bfc.add_conversion(3, 1)  # D -> B
    bfc.add_fusion(0, 1, 2)  # A + B -> C
    bfc.add_fission(2, 3, 1)  # C -> D + B

    # respective backwards reactions
    bfc.add_conversion(1, 3)  # B -> D, nope
    bfc.add_fission(2, 0, 1)  # C -> A + B, nope
    bfc.add_fusion(3, 1, 2)  # D + B -> C, nope

    # some more stuff
    bfc.add_conversion(0, 1)  # A -> B, nope
    bfc.add_conversion(1, 0)  # B -> A, nope

    bfc.add_conversion(0, 2)  # A -> C, nope
    bfc.add_conversion(2, 0)  # C -> A, nope

    bfc.add_conversion(1, 2)  # B -> C, nope
    bfc.add_conversion(2, 1)  # C -> B, nope

    bfc.add_fusion(0, 2, 3)  # A + C -> D, nope
    bfc.add_fission(3, 0, 2)  # D -> A + C, nope

    bfc.add_fusion(0, 3, 2)  # A + D -> C, nope
    bfc.add_fission(2, 0, 3)  # C -> A + D, nope

    assert bfc.n_basis_functions == len(desired_rates), \
        "got {} basis functions but only {} desired rates".format(bfc.n_basis_functions, len(desired_rates))
    return bfc


analysis = ana.ReactionAnalysis(get_bfc(), desired_rates, initial_states, set_up_system,
                                recompute=False, recompute_traj=False, fname_prefix='pw_linear_abcd_',
                                fname_postfix='_250_counts', n_species=4, target_n_counts=250, timestep=5e-4,
                                interp_degree='pw_linear')

gillespie_kw = {'n_steps':220, 'n_realizations':200, 'update_and_persist':False, 'njobs':8, 'atol':1e-9}
analysis.obtain_serialized_gillespie_trajectories(alphas=np.linspace(1, 15000, 8), alpha_search_depth=7,
                                                  **gillespie_kw)