import numpy as np

import readdy_learn.analyze.tools as pat
import readdy_learn.generate.generate_tools.kinetic_monte_carlo as kmc
from readdy_learn.analyze.sklearn import BasisFunctionConfiguration, CV, get_dense_params
from readdy_learn.sample_tools import Suite

import matplotlib.pyplot as plt


def set_up_system(init_state=None):
    if init_state is None:
        init_state = [[70, 30, 15, 10]]
    sys = kmc.ReactionDiffusionSystem(n_species=4, n_boxes=1, diffusivity=[[[0.]], [[0.]], [[0.]], [[0.]]],
                                      init_state=init_state, species_names=["A", "B", "C", "D"])
    sys.add_conversion("A", "D", np.array([4.]))
    sys.add_conversion("D", "A", np.array([0.5]))
    sys.add_fusion("A", "B", "C", np.array([2]))
    sys.add_fission("C", "A", "B", np.array([1.5]))

    bfc = BasisFunctionConfiguration(n_species=sys.n_species)
    bfc.add_conversion(0, 3)  # A -> D
    bfc.add_conversion(3, 0)  # D -> A
    bfc.add_fusion(0, 1, 2)  # A + B -> C
    bfc.add_fission(2, 0, 1)  # C -> A + B
    bfc.add_conversion(0, 1)  # A -> B
    bfc.add_conversion(0, 2)  # A -> C
    return sys, bfc


def get_traj(sys, verbose=True, n_frames=None, timestep=None):
    counts, times, config = sys.get_counts_config(n_frames=n_frames, timestep=timestep)
    traj = pat.Trajectory.from_counts(counts, times[1] - times[0], verbose=verbose)
    traj.update()
    return traj


train_system, bfc = set_up_system()
train_system.simulate(300)
train_traj = get_traj(train_system, timestep=1e-5)

test_trajs = []
for _ in range(1):
    init = [[np.random.randint(0, 70), np.random.randint(0, 70),
             np.random.randint(0, 70), np.random.randint(0, 70)]]
    print("setting up trajectory with init state {}".format(init))
    test_system, _ = set_up_system(init_state=init)
    test_system.simulate(300)
    test_traj = get_traj(test_system, timestep=1e-5)
    test_trajs.append(test_traj)

suite = Suite(set_up_system)
init_rates = suite.calculate("", [1e-3], 300, save=False)

alphas = np.linspace(0, 200, num=4)
l1_ratios = np.array([1.0])
cv = CV(train_traj, bfc, -1, alphas, l1_ratios, 5, init_rates, test_traj=test_trajs, maxiter=300000)
cv.fit_cross_trajs()
