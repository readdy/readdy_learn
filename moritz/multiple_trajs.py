import os
import numpy as np
import matplotlib as mpl

import readdy_learn.analyze.analyze as ana
import readdy_learn.analyze.basis as basis

import pynumtools.kmc as kmc

# mpl.rcParams['figure.figsize'] = (13, 13)
import matplotlib.pyplot as plt
import scipy.signal as ss
from readdy_learn.example.regulation_network import RegulationNetwork

regulation_network = RegulationNetwork()

if __name__ == '__main__':
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    for train, test in loo.split([1,2,3,4]):
        print("train: {}, test: {}".format(train, test))

    if False:
        analysis = regulation_network.generate_analysis_object()
        for i in range(4):
            print("traj {}".format(i))
            t = analysis.generate_or_load_traj_lma(i, regulation_network.target_time,
                                                   noise_variance=regulation_network.noise_variance,
                                                   realizations=regulation_network.realisations)
            t.persist()
            # analysis.plot_and_persist_lma_traj(t)
            regulation_network.compute_gradient_derivatives(analysis)
            # regulation_network.plot_concentrations(analysis, i)
        cv_res = analysis.elastic_net_cv([i for i in range(4)],
                                         alphas=np.concatenate(
                                             (np.linspace(0.000001, 0.0001, num=16),
                                              np.linspace(0.0002, 0.012, num=16),
                                              np.linspace(0.012, 0.1, num=8))),
                                         l1_ratios=np.linspace(0., 1., num=3))
