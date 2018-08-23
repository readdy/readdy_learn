import matplotlib.pyplot as plt
import os
import numpy as np
import case_config

plt.style.use("lm.mplstyle")

#DATA_ROOT = "/home/chris/Dropbox/readdy_learn/reaction_learn_data"
DATA_ROOT = "/srv/public/chrisfr/workspace/data/reaction_learn_data"

traj = case_config.get_traj_from_file(os.path.join(DATA_ROOT, "gillespie_trajs_init_1.h5"), 20000, 0)

#which = "mrna"
#which = "protein"
which = "all"

if which=="mrna":
    species = np.array([1, 4, 7])
    ylim = (-0.05, 0.85)
elif which=="protein":
    species = np.array([2, 5, 8])
    ylim = (-0.05, 0.85)
elif which=="all":
    species = np.array([1, 2, 4, 5, 7, 8])
    ylim = (-0.05, 0.85)
else:
    raise ValueError("blub")

filename = "illustrative_traj_init_1_"+which+".pdf"

plot_format = ["-"] * 6
#plot_format = ["-", "--", ":","-", "--", ":"]

t = np.arange(len(traj[0]))*case_config.TIMESTEP
stride = 1
for ix, s in enumerate(species):
    name = case_config.SPECIES_TEX[s]
    plt.plot(t[::stride], traj[0][:,s][::stride], plot_format[ix], label=r"${}$".format(name), color='C{}'.format(ix))
plt.legend(loc="lower right")
plt.ylim(ylim)
plt.xlabel('Time in a.u.')
plt.ylabel('Concentration in a.u.')
plt.gcf().tight_layout()

plt.savefig(filename)
plt.show()
