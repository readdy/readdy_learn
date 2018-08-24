import matplotlib.pyplot as plt
import os
import numpy as np
import case_config

plt.style.use("rlearn.mplstyle")

DATA_ROOT = "/home/chris/Dropbox/readdy_learn/reaction_learn_data"
# DATA_ROOT = "/srv/public/chrisfr/workspace/data/reaction_learn_data"

traj = case_config.get_traj_from_file(os.path.join(DATA_ROOT, "gillespie_trajs_init_1.h5"), 20000, 0)


def plot_species(species):
    plot_format = ["-", "--"] * 6
    for ix, s in enumerate(species):
        name = case_config.SPECIES_TEX[s]
        plt.plot(t[::stride], traj[0][:, s][::stride], plot_format[ix], label=r"${}$".format(name),
                 #color='C{}'.format(ix)
                 color="black"
                 )


t = np.arange(len(traj[0])) * case_config.TIMESTEP
stride = 1

# width = 7.1 # double columns
width = 7.1 / 2.  # single column
width = 4.5
n_rows = 3
height = n_rows * 1.3
n_cols = 1

fig, axarr = plt.subplots(n_rows, n_cols, figsize=(width, height), sharex=True)
axlist = axarr.flatten()

plt.sca(axlist[0])
plot_species(np.array([2, 1]))
#plt.ylabel('Concentration in a.u.')
plt.legend(loc="center left", bbox_to_anchor=(1,0.5))

plt.sca(axlist[1])
plot_species(np.array([5, 4]))
plt.ylabel('Concentration in a.u.', labelpad=20)
plt.legend(loc="center left", bbox_to_anchor=(1,0.5))

plt.sca(axlist[2])
plot_species(np.array([8, 7]))
plt.xlabel('Time in a.u.')
#plt.ylabel('Concentration in a.u.')
plt.legend(loc="center left", bbox_to_anchor=(1,0.5))

fig.tight_layout()
fig.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

plt.savefig("./illustrative-traj.pdf", transparent=True)
plt.show()
plt.clf()
plt.close(fig)
