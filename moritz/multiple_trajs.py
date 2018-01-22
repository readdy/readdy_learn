import os
import numpy as np
import matplotlib as mpl

import readdy_learn.analyze.analyze as ana
import readdy_learn.analyze.basis as basis

import pynumtools.kmc as kmc

#mpl.rcParams['figure.figsize'] = (13, 13)
import matplotlib.pyplot as plt
import scipy.signal as ss
from readdy_learn.example.regulation_network import RegulationNetwork

regulation_network = RegulationNetwork()

def plot_overview(plot_functions, n_rows=2, n_cols=2, size_factor=1.):
    plt.figure(1, figsize=(5*n_cols*size_factor,3.5*n_rows*size_factor))
    n_plots = n_rows * n_cols
    idx = 1
    for pf in plot_functions:
        if idx > n_plots:
            break
        plt.subplot(n_rows, n_cols, idx)
        pf()
        plt.legend(loc="best")
        idx += 1

    #plt.subplots_adjust(top=0.88, bottom=0.08, left=0.10, right=0.95, hspace=0.4, wspace=0.35)
    plt.tight_layout(pad=0.6, w_pad=2.0, h_pad=2.0)

def plot_and_persist_lma_traj(t):
    plt.plot(t.counts[:, 0], label=regulation_network.species_names[0])
    plt.plot(t.counts[:, 1], label=regulation_network.species_names[1])
    plt.plot(t.counts[:, 2], label=regulation_network.species_names[2])
    plt.plot(t.counts[:, 3], label=regulation_network.species_names[3])
    plt.plot(t.counts[:, 4], label=regulation_network.species_names[4])
    plt.plot(t.counts[:, 5], label=regulation_network.species_names[5])
    plt.plot(t.counts[:, 6], label=regulation_network.species_names[6])
    plt.plot(t.counts[:, 7], label=regulation_network.species_names[7])
    plt.plot(t.counts[:, 8], label=regulation_network.species_names[8])
    plt.legend(loc="best")
    plt.show()
    t.persist()


if __name__ == '__main__':
    analysis = regulation_network.generate_analysis_object()
    for i in range(4):
        print("traj {}".format(i))
        t = analysis.generate_or_load_traj_lma(i, regulation_network.target_time,
                                               noise_variance=regulation_network.noise_variance,
                                               realizations=regulation_network.realisations)
        plot_and_persist_lma_traj(t)
