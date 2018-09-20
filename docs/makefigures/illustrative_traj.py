import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import numpy as np
import case_config

plt.style.use("./rlearn.mplstyle")

DATA_ROOT = "/home/mi/chrisfr/Dropbox/readdy_learn/reaction_learn_data"
# DATA_ROOT = "/srv/public/chrisfr/workspace/data/reaction_learn_data"

_, traj_lma = case_config.integrate(case_config.INITIAL_CONDITIONS[0], case_config.DESIRED_RATES)
traj = case_config.get_traj_from_file(os.path.join(DATA_ROOT, "gillespie_trajs_init_1.h5"), 1, 0)

t = np.arange(len(traj[0])) * case_config.TIMESTEP
stride = 1


def plot_species(ax, species):
    plot_format = ["-", "--"] * 6
    plot_color = ['C0', 'C1'] * 6
    for ix, s in enumerate(species):
        name = case_config.SPECIES_TEX[s]
        ax.plot(t[::stride], traj_lma[:, s][::stride], plot_format[ix], color=plot_color[ix],
                label=r"${}$".format(name))
        # ax.plot(t[::stride], traj[0][:, s][::stride], color='grey', alpha=.6)


fs = plt.rcParams.get('figure.figsize')
f = plt.figure(figsize=(fs[0], 2.5 * fs[1]))

gs = GridSpec(4, 2, height_ratios=[3, 1, 1, 1], width_ratios=[0.05, 0.95])
# leave space above for the scheme
ax1 = plt.subplot(gs[1, 1])
ax2 = plt.subplot(gs[2, 1], sharex=ax1)
ax3 = plt.subplot(gs[3, 1], sharex=ax1)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)

plot_species(ax1, np.array([2, 1]))
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=2)
plot_species(ax2, np.array([5, 4]))
ax2.set_ylabel('Concentration in a.u.', labelpad=10.)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=2)
plot_species(ax3, np.array([8, 7]))
ax3.set_xlabel('Time in a.u.', labelpad=-1)
ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -.07), ncol=2)

f.text(0.05, .86, r'\textbf{(a)}', fontdict={'size': plt.rcParams['font.size'] + 4})
f.text(0.05, .5, r'\textbf{(b)}', fontdict={'size': plt.rcParams['font.size'] + 4})

f.savefig("scheme.pdf", bbox_inches="tight")
plt.show()
