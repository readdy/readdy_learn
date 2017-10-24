import argparse
import os

import numpy as np

import readdy_learn.generate.generate_tools.kinetic_monte_carlo as kmc
from readdy_learn.analyze.sklearn import BasisFunctionConfiguration
from readdy_learn.sample_tools import Suite


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

    suite = Suite(set_up_system, interp_degree=5)

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfile", help="the outfile", type=str)
    parser.add_argument("-f", "--force", help="perform it already!", action="store_true")
    parser.add_argument("-p", "--plot", help="just plot the data", action="store_true")
    parser.add_argument("--n_steps", help="the number of gillespie steps", type=int)
    parser.add_argument("--n_realizations", help="the number of realizations, defaults to 20", type=int)
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
        suite.plot(outfile)
    else:
        if not args.n_steps:
            n_steps = 300
        else:
            n_steps = args.n_steps

        if not args.n_realizations:
            n_realizations = 20
        else:
            n_realizations = args.n_realizations

        print("---> running analysis for n_steps={} with n_realizations={}".format(n_steps, n_realizations))
        timesteps = [.0001] + [x for x in np.arange(.001, .1, step=.001)]
        suite.calculate(outfile, timesteps=timesteps, n_steps=n_steps, n_realizations=n_realizations,
                        verbose=args.verbose, write_concentrations_for_time_step=.001)
