import argparse

import numpy as np
import os
import readdy_learn.generate.generate_tools.kinetic_monte_carlo as kmc
import readdy_learn.analyze.tools as pat
from readdy_learn.analyze.sklearn import BasisFunctionConfiguration
from readdy_learn.analyze.sklearn import ReaDDyElasticNetEstimator
from scipy.integrate import odeint

import matplotlib.pyplot as plt


def run(sys, bfc, verbose=True, n_frames=None, timestep=None):
    counts, times, config = sys.get_counts_config(n_frames=n_frames, timestep=timestep)

    traj = pat.Trajectory.from_counts(config, counts, times[1] - times[0])
    traj.update()

    est = ReaDDyElasticNetEstimator(traj, bfc, scale=-1, alpha=0., l1_ratio=1., method='SLSQP',
                                    verbose=verbose, approx_jac=False, options={'ftol': 1e-16})
    est.fit(None)
    coefficients = est.coefficients_
    return coefficients


def estimated_behavior(coefficients, bfc, initial_counts, time_step, n_time_steps):
    def fun(data, _):
        theta = np.array([f(data) for f in bfc.functions])
        return np.matmul(coefficients, theta)

    estimated_realisation = odeint(fun, initial_counts,
                                   np.arange(0., n_time_steps * time_step, time_step))
    return estimated_realisation


def plot(file):
    system, bfc = set_up_system()
    config = system.get_trajectory_config()

    f = np.load(file)
    data = f['rates'].item()
    counts = f['counts']
    times = f['times']
    xs = np.asarray([k for k in data.keys()])
    smallest_time_step = min(data.keys())

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    estimated = estimated_behavior(np.mean(data[smallest_time_step], axis=0), bfc, counts[0], smallest_time_step,
                                   len(times))

    for t in config.types.keys():
        type_id = config.types[t]
        ax1.plot(times, estimated[:, type_id], "k--")
        ax1.plot(times, counts[:, type_id], label="counts " + t)
    ax1.legend(loc="best")

    ax2.set_title('Estimated rates')
    for i, reaction in enumerate(system.reactions):
        ys, yerr = [], []
        for time_step in data.keys():
            rates = data[time_step]
            ys.append(np.mean(rates[:, i]))
            yerr.append(np.std(rates[:, i]))
        ax2.errorbar(xs, ys, yerr=yerr, label='estimated ' + str(reaction))
        ax2.plot(xs, reaction.rate * np.ones_like(xs), "--", label="expected " + str(reaction))

    # ax2.set_xscale('log')
    ax2.set_xlabel("time step")
    ax2.set_ylabel("rate")
    ax2.legend(loc="best")

    fig.show()
    plt.show()


def calculate(file, timesteps, n_steps, write_concentrations_for_time_step=None, verbose=True):
    if os.path.exists(file):
        raise ValueError("File already existed: {}".format(file))

    allrates = {}

    for k in timesteps:
        allrates[k] = []

    if write_concentrations_for_time_step is None:
        write_concentrations_for_time_step = min(timesteps)
    concentrations = None

    for n in range(20):
        system, bfc = set_up_system()
        system.simulate(n_steps)
        for dt in timesteps:
            rates = run(system, bfc, timestep=dt, verbose=verbose)
            allrates[dt].append(rates)
            if dt == write_concentrations_for_time_step:
                counts, times, config = system.get_counts_config(timestep=dt)
                concentrations = counts.squeeze(), times.squeeze()
    for k in allrates.keys():
        allrates[k] = np.asarray(allrates[k])
    np.savez(file, rates=allrates, counts=concentrations[0], times=concentrations[1])
    for dt in timesteps:
        rates = allrates[dt]
        if verbose:
            print("got {:.3f}±{:.3f}  and {:.3f}±{:.3f} for timestep={}".format(
                np.mean(rates[:, 0]), np.std(rates[:, 0]),
                np.mean(rates[:, 1]), np.std(rates[:, 1]), dt))


def set_up_system():
    sys = kmc.ReactionDiffusionSystem(n_species=2, n_boxes=1, diffusivity=[[[0.]], [[0.]]], init_state=[[70, 0]],
                                      species_names=["A", "B"])
    sys.add_conversion("A", "B", np.array([4.]))
    sys.add_conversion("B", "A", np.array([0.5]))

    bfc = BasisFunctionConfiguration(n_species=sys.n_species)
    bfc.add_conversion(0, 1)  # A -> B
    bfc.add_conversion(1, 0)  # B -> A
    return sys, bfc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfile", help="the outfile", type=str)
    parser.add_argument("-f", "--force", help="perform it already!", action="store_true")
    parser.add_argument("-p", "--plot", help="just plot the data", action="store_true")
    parser.add_argument("--n_steps", help="the number of gillespie steps", type=int)
    parser.add_argument("-v", "--verbose", help="verbose output", action="store_true")
    args = parser.parse_args()
    if not args.outfile:
        outfile = 'convergence_simple.npz'
        print("---> using default outfile {}".format(outfile))
    else:
        outfile = args.outfile
        print("---> using custom outfile {}".format(outfile))

    if args.force:
        print("---> got the force argument, remove output file if existing and proceed")
        if os.path.exists(outfile):
            os.remove(outfile)

    if args.plot:
        print("---> got plot argument, just plotting the outfile if it exists")
        plot(outfile)
    else:
        if not args.n_steps:
            n_steps = 300
        else:
            n_steps = args.n_steps

        print("---> running analysis for n_steps={}".format(n_steps))
        timesteps = [.000001, .00001, .0001] + [x for x in np.arange(.001, .5, step=.005)]
        calculate(outfile, timesteps=timesteps, n_steps=n_steps, verbose=args.verbose)
