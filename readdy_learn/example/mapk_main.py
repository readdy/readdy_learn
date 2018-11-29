import os
from copy import copy

import numpy as np
import pickle
import argparse

import readdy_learn.example.mapk as mapk

parser = argparse.ArgumentParser()
parser.add_argument("--ix", type=int, required=True)
parser.add_argument("--outdir", type=str, required=True)

begins = np.arange(0, mapk.n_combinations(), 10)
ends = begins + 10
ends[-1] = np.min((ends[-1], mapk.n_combinations()))

if __name__ == '__main__':

    args = parser.parse_args()

    begin = begins[args.ix]
    end = ends[args.ix]

    j = 0
    for selection in mapk.conversion_ops_range(begin, end):

        conf = mapk.MAPKConfiguration(selection)

        data = []
        for N_STIMULUS in [1e2, 1e3, 1e4]:
            mapk.TIMESTEP = 1e-5

            X = copy(mapk.INITIAL_STATES)
            X[0][0] = N_STIMULUS
            mapk.INITIAL_STATES = X
            time, count, dcount_dt = mapk.generate_lma(.003, config=conf)
            data.append((time, count, dcount_dt))

        time = np.concatenate([data[i][0] for i in range(len(data))])
        counts = np.concatenate([data[i][1] for i in range(len(data))])
        dcounts_dt = np.concatenate([data[i][2] for i in range(len(data))])

        result = mapk.solve_grid(conf, counts, dcounts_dt, l1_ratios=(1.,),
                                 alphas=np.logspace(-7, -1, num=23), njobs=1)

        alphas = []
        l1_ratios = []
        estimated_rates = []
        for res in result:
            if all(np.isfinite(res[2])):
                alphas.append(res[0])
                l1_ratios.append(res[1])
                estimated_rates.append(res[2])
        alphas = np.array(alphas)
        l1_ratios = np.array(l1_ratios)
        estimated_rates = np.array(estimated_rates)

        sort = np.argsort(alphas)
        alphas = alphas[sort]
        l1_ratios = l1_ratios[sort]
        estimated_rates = estimated_rates[sort]
        l1_errs = np.sum(np.abs(estimated_rates - conf.rates), axis=1)

        l1_min_ix = np.argmin(l1_errs)

        outfile = os.path.join(args.outdir, f'{begin+j}.pkl')

        if os.path.exists(outfile):
            os.remove(outfile)

        with open(outfile, 'wb') as f:
            pickle.dump({
                'alphas': alphas,
                'l1_errs': l1_errs,
                'estimated_rates': estimated_rates,
                'l1_ratios': l1_ratios,
                'best_l1_err': l1_errs[l1_min_ix],
                'best_alpha': alphas[l1_min_ix],
                'best_ix': l1_min_ix,
                'begin': begin,
                'end': end,
                'global_index': begin+j,
                'selection': selection
            }, f)
        # np.savez(outfile, alphas=alphas, l1_errs=l1_errs, estimated_rates=estimated_rates, l1_ratios=l1_ratios)
        j += 1
