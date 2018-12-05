import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator
import numpy as np

if __name__ == '__main__':
    plt.style.use("./rlearn.mplstyle")

    # MAPK
    print(5 * "-----" + "MAPK" + 5 * "-----")
    mapk_file = "../../mapk_pathway/mapk_data.npz"
    with np.load(mapk_file) as f:
        time = f["time"]
        counts = f["counts"]
        dcounts_dt = f["dcounts_dt"]
        alphas_gs = f["alphas_gs"]
        l1_errs_gs = f["l1_errs_gs"]
        estimated_rates_gs = f["estimated_rates_gs"]
        best_estimated_rates_gs = f["best_estimated_rates_gs"]
        alphas_cv = f["alphas_cv"]
        scores_cv = f["scores_cv"]
        rates_cv = f["rates_cv"]
        rates_lsq = f["rates_lsq"]
        rates_desired = f["rates_desired"]
        activities = f["activities"]
        stimulus = f["stimulus"]
        init_state_activities = f["init_state_activities"]

    # plot data ?
    print(f"Number of frames: {len(time)}")


    # plot cross validation scores
    sel = np.argmin(-scores_cv)
    plt.semilogx(alphas_cv, -scores_cv)
    #plt.plot([alphas_cv[sel]], [-scores_cv[sel]], 'x', markersize=20)
    print(f"Hyperparameter alpha {alphas_cv[sel]}")
    plt.ylim(0., 0.5e-9)
    xmin, xmax = plt.xlim()
    plt.xlim(xmin, 2e-8)
    plt.xlabel(r"Hyperparameter $\alpha$")
    plt.ylabel(r"Mean loss in val. set")
    plt.gcf().tight_layout()
    plt.savefig("mapk_cross_validation.pdf", bbox_inches="tight")
    #plt.show()
    plt.clf()

    def plot_mapk_activation_curve():
        plt.semilogx(stimulus, activities, label="MAPK cascade response")
        plt.xlabel(r'Stimulus $[\mathrm{S}]$ in a.u.')
        plt.ylabel(r'Activity $[\mathrm{TF*}]$ in a.u.')
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, 0.35)
        plt.vlines(init_state_activities, *plt.ylim(), color="xkcd:red", linestyle="--",label="initial conditions", lw=1)
        plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        plt.legend(loc="upper left")


    def plot_mapk_models():
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter

        majorLocator = MultipleLocator(3)
        minorLocator = MultipleLocator(1)
        x_values = [r"$\theta_{%s}$" % (i+1) for i in range(len(rates_lsq))]
        plt.plot(rates_cv, 'o', label="regularized")
        plt.plot(rates_lsq, 'd', label="LSQ")
        # plt.vlines([7.5], 0, 1, 'grey', 'dashed')
        plt.plot(rates_desired, 'x', label="ground truth")
        #plt.xticks([0, 5, 10], [r'$a^2$', r'$b^2$', r'$c^2$'])
        labels = [x for (i,x) in enumerate(x_values) if i % 3 == 0]  # list(map(str, range(1, len(rates_desired) + 1)))
        #for i in range(len(labels)):
        #    if i % 3 == 0:
        #        labels[i] = x_values[i]
        #    else:
        #        labels[i] = ""
        plt.gca().xaxis.set_major_locator(majorLocator)
        plt.gca().xaxis.set_minor_locator(minorLocator)
        plt.xticks(ticks=[i*3 for i in range(len(labels))], labels=labels, fontsize=11)
        plt.xlim([-.5, None])
        plt.legend()
        plt.xlabel(r"Ansatz reaction $\theta_r$")
        plt.ylabel("Rate constant in a.u.")

    # Combined figure of activation and results
    fs = plt.rcParams.get('figure.figsize')
    figure = plt.figure(figsize=(fs[0], 2. * fs[1]))

    gs = GridSpec(2, 1)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])

    plt.sca(ax1)
    plot_mapk_activation_curve()

    plt.sca(ax2)
    plot_mapk_models()

    figure.text(0., .96, r'\textbf{(a)}', fontdict={'size': plt.rcParams['font.size'] + 4})
    figure.text(0., .48, r'\textbf{(b)}', fontdict={'size': plt.rcParams['font.size'] + 4})

    figure.tight_layout()
    figure.savefig("mapk.pdf", bbox_inches="tight", transparent=True, dpi=300)

    #plt.show()
    plt.clf()


    print("1-norm of relative deviation from the ground truth rates (LSQ):", np.sum(np.abs(((rates_lsq[:8] - rates_desired[:8]) / rates_desired[:8]))))

    print("1-norm of relative deviation from the ground truth rates (regularized):", np.sum(np.abs(((rates_cv[:8] - rates_desired[:8]) / rates_desired[:8]))))


    # Predator Prey
    print(5 * "-----" + "Predator prey" + 5 * "-----")
    pprey_file = "../../predator_prey/predator_prey_data.npz"
    with np.load(pprey_file) as f:
        time = f["time"]
        counts = f["counts"]
        dcounts_dt = f["dcounts_dt"]
        alphas_gs = f["alphas_gs"]
        l1_errs_gs = f["l1_errs_gs"]
        estimated_rates_gs = f["estimated_rates_gs"]
        best_estimated_rates_gs = f["best_estimated_rates_gs"]
        alphas_cv = f["alphas_cv"]
        scores_cv = f["scores_cv"]
        rates_cv = f["rates_cv"]
        rates_lsq = f["rates_lsq"]
        rates_desired = f["rates_desired"]

    print(f"Number of frames: {len(time)}")

    def plot_pprey_traj():
        plt.plot(time, counts[:, 0], label='prey')
        plt.plot(time, counts[:, 1], label='predator')
        plt.xlabel("Time in a.u.")
        plt.ylabel("Concentration in a.u.")
        plt.legend()

    def plot_pprey_models():
        plt.plot(rates_cv, 'o', label="regularized")
        plt.plot(rates_lsq, 'd', label="LSQ")
        # plt.vlines([7.5], 0, 1, 'grey', 'dashed')
        plt.plot(rates_desired, 'x', label="ground truth")
        labels = list(map(str, range(1, len(rates_desired) + 1)))
        for i in range(len(labels)):
            if i % 2 == 0:
                pass
            else:
                labels[i] = ""
        plt.xticks(ticks=range(len(rates_desired)), labels=labels, fontsize=11)
        plt.legend()
        plt.xlabel(r"Ansatz reaction \#")
        plt.ylabel("Rate constant in a.u.")

    # Combined figure of traj and result
    fs = plt.rcParams.get('figure.figsize')
    figure = plt.figure(figsize=(fs[0], 2. * fs[1]))

    gs = GridSpec(2, 1)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])

    plt.sca(ax1)
    plot_pprey_traj()

    plt.sca(ax2)
    plot_pprey_models()

    figure.text(0., .96, r'\textbf{(a)}', fontdict={'size': plt.rcParams['font.size'] + 4})
    figure.text(0., .48, r'\textbf{(b)}', fontdict={'size': plt.rcParams['font.size'] + 4})

    figure.tight_layout()
    figure.savefig("pprey.pdf", bbox_inches="tight", transparent=True, dpi=300)

    # plot cross validation scores
    sel = np.argmin(-scores_cv)
    plt.semilogx(alphas_cv, -scores_cv)
    #plt.plot([alphas_cv[sel]], [-scores_cv[sel]], 'x', markersize=20)
    print(f"Hyperparameter alpha {alphas_cv[sel]}")
    plt.ylim(6.45e-7, 6.7e-7)
    xmin, xmax = plt.xlim()
    plt.xlim(1e-9, 1e-5)
    plt.xlabel(r"Hyperparameter $\alpha$")
    plt.ylabel(r"Mean loss in val. set")
    plt.gcf().tight_layout()
    plt.savefig("pprey_cross_validation.pdf", bbox_inches="tight")
    #plt.show()
    plt.clf()



    print("1-norm of relative deviation from the ground truth rates (LSQ):", np.sum(np.abs(((rates_lsq[:5] - rates_desired[:5]) / rates_desired[:5]))))

    print("1-norm of relative deviation from the ground truth rates (regularized):", np.sum(np.abs(((rates_cv[:5] - rates_desired[:5]) / rates_desired[:5]))))
