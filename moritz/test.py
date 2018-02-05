import numpy as np

import readdy_learn.example.regulation_network as regulation_network
import readdy_learn.analyze.validation as validation

import matplotlib.pyplot as plt

rn = regulation_network.RegulationNetwork()
rn.noise_variance = .01
rn.timestep = 1e-0
val = validation.Validation(rn, True, 6)

result = val.validate(np.logspace(-6, -1, num=4), lambdas=np.linspace(0, 1, num=2), realizations=3)
print(result)

def get_n_realizations(result):
    some_alpha = result[0]['alpha']
    some_l1_ratio = result[0]['l1_ratio']
    some_cutoff = result[0]['cutoff']
    count = 0
    for r in result:
        if r['alpha'] == some_alpha and r['l1_ratio'] == some_l1_ratio and r['cutoff'] == some_cutoff:
            count += 1
    return count


def get_distinct_l1_ratios(result):
    s = set()
    for r in result:
        s.add(r['l1_ratio'])
    return sorted(list(s))


def get_distinct_alphas(result):
    s = set()
    for r in result:
        s.add(r['alpha'])
    return sorted(list(s))


def get_scores(result, alpha, l1_ratio):
    return [r['score'] for r in result if r['alpha'] == alpha and r['l1_ratio'] == l1_ratio]


l1_ratios = get_distinct_l1_ratios(result)
alphas = get_distinct_alphas(result)
for l1_ratio in l1_ratios:
    xs = np.array(alphas)
    ys = np.empty_like(xs)
    yerr = np.empty_like(xs)
    for ix, alpha in enumerate(alphas):
        ys[ix] = np.mean(get_scores(result, alpha, l1_ratio))
        yerr[ix] = np.std(get_scores(result, alpha, l1_ratio))
    print(yerr)
    plt.errorbar(xs, ys, yerr=yerr)
plt.show()
