import numpy as np
import os
import readdy_learn.generate.generate_tools.kinetic_monte_carlo as kmc
import readdy_learn.analyze.tools as pat
from readdy_learn.analyze.sklearn import BasisFunctionConfiguration
from readdy_learn.analyze.sklearn import ReaDDyElasticNetEstimator

import matplotlib.pyplot as plt


def simulate(n_steps):
    sys = kmc.ReactionDiffusionSystem(n_species=2, n_boxes=1, diffusivity=[[[0.]], [[0.]]], init_state=[[70, 0]],
                                      species_names=["A", "B"])
    sys.add_conversion("A", "B", np.array([4.]))
    sys.add_conversion("B", "A", np.array([0.5]))
    sys.simulate(n_steps)
    return sys


def run(sys, n_frames=None, timestep=None):
    counts, times, config = sys.get_counts_config(n_frames=n_frames, timestep=timestep)

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


def plot(file):
    f = np.load(file)
    data = f['rates'].item()
    counts = f['counts']
    times = f['times']
    xs = np.asarray([k for k in data.keys()])
    ys1, yerr1 = [], []
    ys2, yerr2 = [], []
    for time_step in data.keys():
        rates = data[time_step]
        ys1.append(np.mean(rates[:, 0]))
        yerr1.append(np.std(rates[:, 0]))
        ys2.append(np.mean(rates[:, 1]))
        yerr2.append(np.std(rates[:, 1]))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.set_title('Concentration curves')
    ax1.plot(times, counts[:, 0], label='Counts A')
    ax1.plot(times, counts[:, 1], label='Counts B')
    ax1.set_xlabel('time')
    ax1.set_ylabel('counts')
    ax1.legend(loc='best')

    ax2.set_title('Estimated rates')
    ax2.errorbar(xs, ys1, yerr=yerr1, label='estimated A->B')
    ax2.plot(xs, 4.*np.ones_like(xs), "--", label="expected A->B")
    ax2.errorbar(xs, ys2, yerr=yerr2, label='estimated B->A')
    ax2.plot(xs, .5* np.ones_like(xs), "--", label="expected B->A")
    ax2.set_xlabel("time step")
    ax2.set_ylabel("rate")
    ax2.legend(loc="best")

    fig.show()
    plt.show()


def calculate(file, write_concentrations_for_time_step=None):
    if os.path.exists(file):
        raise ValueError("File already existed: {}".format(file))

    allrates = {}
    timesteps = [.000001, .00001, .0001] + [x for x in np.arange(.001, .5, step=.005)]
    for k in timesteps:
        allrates[k] = []

    if write_concentrations_for_time_step is None:
        write_concentrations_for_time_step = min(timesteps)
    concentrations = None

    for n in range(20):
        system = simulate(300)
        for dt in timesteps:
            rates = run(system, timestep=dt)
            allrates[dt].append(rates)
            if dt == write_concentrations_for_time_step:
                counts, times, config = system.get_counts_config(timestep=dt)
                concentrations = counts.squeeze(), times.squeeze()
    for k in allrates.keys():
        allrates[k] = np.asarray(allrates[k])
    np.savez(file, rates=allrates, counts=concentrations[0], times=concentrations[1])
    for dt in timesteps:
        rates = allrates[dt]
        print("got {:.3f}±{:.3f}  and {:.3f}±{:.3f} for timestep={}".format(
            np.mean(rates[:, 0]), np.std(rates[:, 0]),
            np.mean(rates[:, 1]), np.std(rates[:, 1]), dt))


if __name__ == '__main__':
    outfile = 'convergence_simple.npz'
    #if os.path.exists(outfile):
    #    os.remove(outfile)
    calculate(outfile)
    # outfile = "/home/mho/platypus/Development/readdy_learn/convergence_simple.npy"
    # plot(outfile)
