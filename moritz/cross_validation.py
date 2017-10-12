import argparse
import os

import numpy as np

import readdy_learn.generate.generate_tools.kinetic_monte_carlo as kmc
from readdy_learn.analyze.sklearn import BasisFunctionConfiguration, CV, get_dense_params
from readdy_learn.sample_tools import Suite


def set_up_system():
    sys = kmc.ReactionDiffusionSystem(n_species=4, n_boxes=1, diffusivity=[[[0.]], [[0.]], [[0.]], [[0.]]],
                                      init_state=[[70, 0]], species_names=["A", "B", "C", "D"])
    sys.add_conversion("A", "D", np.array([4.]))
    sys.add_conversion("D", "A", np.array([0.5]))
    sys.add_fusion("A", "B", "C", np.array([2]))
    sys.add_fission("C", "A", "B", np.array([1.5]))

    bfc = BasisFunctionConfiguration(n_species=sys.n_species)
    bfc.add_conversion(0, 3)  # A -> D
    bfc.add_conversion(3, 0)  # D -> A
    bfc.add_fusion(0, 1, 2) # A + B -> C
    bfc.add_fission(2, 0, 1) # C -> A + B
    return sys, bfc


if __name__ == '__main__':

    suite = Suite(set_up_system)

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
        timesteps = [.000001, .00001, .0001] + [x for x in np.arange(.001, .5, step=.005)]
        suite.calculate(outfile, timesteps=timesteps, n_steps=n_steps, n_realizations=n_realizations,
                        verbose=args.verbose, write_concentrations_for_time_step=.001)
