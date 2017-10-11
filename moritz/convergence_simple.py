import numpy as np
import readdy_learn.generate.generate_tools.kinetic_monte_carlo as kmc
import readdy_learn.analyze.tools as pat
from readdy_learn.analyze.sklearn import BasisFunctionConfiguration
from readdy_learn.analyze.sklearn import ReaDDyElasticNetEstimator

import matplotlib.pyplot as plt

def run(n_steps, n_frames=None, timestep=None):
    system = kmc.ReactionDiffusionSystem(n_species=2, n_boxes=1, diffusivity=[[[0.]], [[0.]]], init_state=[[70, 0]],
                                         species_names=["A", "B"])
    system.add_conversion("A", "B", np.array([4.]))
    system.add_conversion("B", "A", np.array([0.5]))
    system.simulate(n_steps)

    counts, times, config = system.get_counts_config(n_frames=n_frames, timestep=timestep)

    traj = pat.Trajectory.from_counts(config, counts, times[1] - times[0])
    traj.update()

    bfc = BasisFunctionConfiguration(n_species=traj.n_species)
    bfc.add_conversion(0, 1)  # A -> B
    bfc.add_conversion(1, 0)  # B -> A
    est = ReaDDyElasticNetEstimator(traj, bfc, scale=-1, alpha=0., l1_ratio=1., method='SLSQP',
                                    verbose=False, approx_jac=False, options={'ftol': 1e-6})
    est.fit(None)
    coefficients = est.coefficients_
    return coefficients

if __name__ == '__main__':
    allrates = []
    timesteps = [.02, .01, .007, .003, .002, .001]
    for dt in timesteps:
        rates = []
        for n in range(100):
            rates.append(run(500, timestep=dt))
        rates = np.asarray(rates).squeeze()
        allrates.append(rates)
        # print(rates)
    for rates, dt in zip(allrates, timesteps):
        print("got {} and {} for timestep={}".format(np.mean(rates[:, 0]), np.mean(rates[:, 1]), dt))
