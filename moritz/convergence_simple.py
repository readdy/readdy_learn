import numpy as np
import os
import readdy_learn.generate.generate_tools.kinetic_monte_carlo as kmc
import readdy_learn.analyze.tools as pat
from readdy_learn.analyze.sklearn import BasisFunctionConfiguration
from readdy_learn.analyze.sklearn import ReaDDyElasticNetEstimator

import matplotlib.pyplot as plt


def simulate(n_steps):
    system = kmc.ReactionDiffusionSystem(n_species=2, n_boxes=1, diffusivity=[[[0.]], [[0.]]], init_state=[[70, 0]],
                                         species_names=["A", "B"])
    system.add_conversion("A", "B", np.array([4.]))
    system.add_conversion("B", "A", np.array([0.5]))
    system.simulate(n_steps)
    return system


def run(system, n_frames=None, timestep=None):
    counts, times, config = system.get_counts_config(n_frames=n_frames, timestep=timestep)

    traj = pat.Trajectory.from_counts(config, counts, times[1] - times[0])
    traj.update()

    bfc = BasisFunctionConfiguration(n_species=traj.n_species)
    bfc.add_conversion(0, 1)  # A -> B
    bfc.add_conversion(1, 0)  # B -> A
    est = ReaDDyElasticNetEstimator(traj, bfc, scale=-1, alpha=0., l1_ratio=1., method='SLSQP',
                                    verbose=True, approx_jac=False, options={'ftol': 1e-16})
    est.fit(None)
    coefficients = est.coefficients_
    return coefficients


if __name__ == '__main__':

    outfile = "convergence_simple.npy"

    if os.path.exists(outfile):
        raise ValueError("File already existed: {}".format(outfile))

    allrates = {}
    timesteps = [.000001, .00001, .0001] + [x for x in np.arange(.001, .5, step=.005)]
    for k in timesteps:
        allrates[k] = []

    for n in range(20):
        system = simulate(300)
        for dt in timesteps:
            rates = run(system, timestep=dt)
            allrates[dt].append(rates)
    for k in allrates.keys():
        allrates[k] = np.asarray(allrates[k])
    np.save(outfile, allrates)
    for dt in timesteps:
        rates = allrates[dt]
        print("got {:.3f}±{:.3f}  and {:.3f}±{:.3f} for timestep={}".format(
            np.mean(rates[:, 0]), np.std(rates[:, 0]),
            np.mean(rates[:, 1]), np.std(rates[:, 1]), dt))
