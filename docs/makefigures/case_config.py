import numpy as np
import h5py as h5
import readdy_learn.analyze.basis as basis
from readdy_learn.example.regulation_network import RegulationNetwork
from tabulate import tabulate

SPECIES = ["DNA_A", "mRNA_A", "A", "DNA_B", "mRNA_B", "B", "DNA_C", "mRNA_C", "C"]
SPECIES_TEX = {
    -1: '\emptyset',
    0: '\mathrm{DNA}_\mathrm{A}',
    1: '\mathrm{mRNA}_\mathrm{A}',
    2: '\mathrm{A}',
    3: '\mathrm{DNA}_\mathrm{B}',
    4: '\mathrm{mRNA}_\mathrm{B}',
    5: '\mathrm{B}',
    6: '\mathrm{DNA}_\mathrm{C}',
    7: '\mathrm{mRNA}_\mathrm{C}',
    8: '\mathrm{C}'
}
CASE1_CUTOFF = 0.22
CASE1_MIN_L1_ERR = 2.4077475813223574
N_ADDITIONAL_FUNS = 2
TARGET_TIME = 2.
SCALE = 500.
TIMESTEP = 3e-3
INITIAL_CONDITIONS = [[1, 2, 0, 1, 0, 3, 1, 0, 0], [1, 1.5, 0, 1, 0, 2., 1, 0, 1.]]

DESIRED_RATES = np.array([
    1.8,  # DA -> DA + MA, transcription A
    2.1,  # MA -> MA + A, translation A
    1.3,  # MA -> 0, decay
    1.5,  # A -> 0, decay
    2.2,  # DB -> DB + MB, transcription B
    2.0,  # MB -> MB + B, translation B
    2.0,  # MB -> 0, decay
    2.5,  # B -> 0, decay
    3.2,  # DC -> DC + MC, transcription C
    3.0,  # MC -> MC + C, translation C
    2.3,  # MC -> 0, decay
    2.5,  # C -> 0, decay
    # self regulation
    0.,  # MA + A -> A, A regulates A
    0.,  # MB + B -> B, B regulates B
    0.,  # MC + C -> C, C regulates C
    # cyclic forward
    0.,  # MB + A -> A, A regulates B
    0.,  # MC + B -> B, B regulates C
    0.,  # MA + C -> C, C regulates A
    # cyclic backward
    6.,  # MC + A -> A, A regulates C
    4.,  # MB + C -> C, C regulates B
    3.,  # MA + B -> B, B regulates A
    # nonsense reactions, mRNA eats protein self
    0., 0., 0.,
    # nonsense reactions, mRNA eats protein cyclic forward
    0., 0., 0.,
    # nonsense reactions, mRNA eats protein  cyclic backward
    0., 0., 0.,
    # nonsense reactions, protein eats protein self
    0., 0., 0.,
    # nonsense reactions, protein eats protein cyclic forward
    0., 0., 0.,
    # nonsense reactions, protein eats protein cyclic backward
    0., 0., 0.,
    # nonsense reactions, protein becomes protein cyclic forward
    0., 0., 0.,
    # nonsense reactions, protein becomes protein cyclic backward
    0., 0., 0.,
])
DESIRED_RATES = np.append(DESIRED_RATES, np.zeros((N_ADDITIONAL_FUNS,)))

"""
# Case2/3, trajs and cv

Trajectories:
    - generate gillespie trajectories with high number of particles, 
    to yield a concentration trajectory with intrinsic chemical noise
    - to control the noise level we average several of such gillespie trajectories
    - TARGET_TIME, TIMESTEP and SCALE parameters as defined above
    - gillespie_realisations = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    - 10 independently generated trajs for each gillespie_realisation

Cross validation using kfold splitter with a number of folds, given one hyperparameter learn the model on one subset of the data, and calculate the validation loss of this model against the remaining data. This is done for all subsets of the splitting. From these losses we drop the first fold and average over the remaining. This is repeated for a range of hyperparameters, yielding validation loss as a function of the hyperparameter. This typically shows a minimum, which we find by a grid-search. This is the hyperparameter we eventually use to estimate the final model on all data in the trajectory.

Case 2 (allegro run 19):
    - traj from one initial condition, INITIAL_CONDITIONS[0], which is RegulationNetwork().initial_states[1]
    - alphas = np.logspace(-8, -2, num=50)
    - n_folds = 10

Case 3 (allegro run 21):
    - traj from two initial conditions, INITIAL_CONDITIONS[0] and INITIAL_CONDITIONS[1], 
    which are RegulationNetwork().initial_states[1] and RegulationNetwork().initial_states[3]
    - the trajectories from the two initial conditions are simply concatenated
    - alphas = np.logspace(-6, 0, num=50)
    - n_folds = 20

"""

def failure_rate(estimated_rates):
    mask = np.zeros_like(DESIRED_RATES)
    active_processes = np.where(DESIRED_RATES > 0)
    mask[active_processes] = 1.

    estimated_mask = np.zeros_like(DESIRED_RATES)
    estimated_mask[np.where(estimated_rates >= CASE1_CUTOFF)] = 1.

    diff = np.sum(np.abs(mask - estimated_mask))
    return diff


def get_bfc_custom():
    # species DA  MA  A  DB  MB  B  DC  MC  C
    # ids     0   1   2  3   4   5  6   7   8
    bfc = basis.BasisFunctionConfiguration(9)
    # usual stuff A
    bfc.add_fission(0, 0, 1)  # 0   DA -> DA + MA, transcription A
    bfc.add_fission(1, 1, 2)  # 1   MA -> MA + A, translation A
    bfc.add_decay(1)  # 2   MA -> 0, decay
    bfc.add_decay(2)  # 3   A -> 0, decay
    # usual stuff B
    bfc.add_fission(3, 3, 4)  # 4   DB -> DB + MB, transcription B
    bfc.add_fission(4, 4, 5)  # 5   MB -> MB + B, translation B
    bfc.add_decay(4)  # 6   MB -> 0, decay
    bfc.add_decay(5)  # 7   B -> 0, decay
    # usual stuff C
    bfc.add_fission(6, 6, 7)  # 8   DC -> DC + MC, transcription C
    bfc.add_fission(7, 7, 8)  # 9   MC -> MC + C, translation C
    bfc.add_decay(7)  # 10  MC -> 0, decay
    bfc.add_decay(8)  # 11  C -> 0, decay

    # all possible regulations
    # self regulation
    bfc.add_fusion(1, 2, 2)  # 12  MA + A -> A, A regulates A
    bfc.add_fusion(4, 5, 5)  # 13  MB + B -> B, B regulates B
    bfc.add_fusion(7, 8, 8)  # 14  MC + C -> C, C regulates C
    # cyclic forward
    bfc.add_fusion(4, 2, 2)  # 15  MB + A -> A, A regulates B
    bfc.add_fusion(7, 5, 5)  # 16  MC + B -> B, B regulates C
    bfc.add_fusion(1, 8, 8)  # 17  MA + C -> C, C regulates A
    # cyclic backward
    bfc.add_fusion(7, 2, 2)  # 18  MC + A -> A, A regulates C
    bfc.add_fusion(4, 8, 8)  # 19  MB + C -> C, C regulates B
    bfc.add_fusion(1, 5, 5)  # 20  MA + B -> B, B regulates A

    # thrown these out due to being identical to decay
    # nonsense reactions, DNA eats mRNA self
    # bfc.add_fusion(1, 0, 0) # 21 MA + DA -> DA
    # bfc.add_fusion(4, 3, 3) # 22 MB + DB -> DB
    # bfc.add_fusion(7, 6, 6) # 23 MC + DC -> DC

    # nonsense reactions, DNA eats mRNA cyclic forward
    # bfc.add_fusion(4, 0, 0) # 24 MB + DA -> DA
    # bfc.add_fusion(7, 3, 3) # 25 MC + DB -> DB
    # bfc.add_fusion(1, 6, 6) # 26 MA + DC -> DC

    # nonsense reactions, DNA eats mRNA cyclic backward
    # bfc.add_fusion(7, 0, 0) # 27 MC + DA -> DA
    # bfc.add_fusion(4, 6, 6) # 28 MB + DC -> DC
    # bfc.add_fusion(1, 3, 3) # 29 MA + DB -> DB

    # nonsense reactions, mRNA eats protein self
    bfc.add_fusion(1, 2, 1)  # 21 MA + A -> MA
    bfc.add_fusion(4, 5, 4)  # 22 MB + B -> MB
    bfc.add_fusion(7, 8, 8)  # 23 MC + C -> MC

    # nonsense reactions, mRNA eats protein cyclic forward
    bfc.add_fusion(1, 5, 1)  # 24 MA + B -> MA
    bfc.add_fusion(4, 8, 4)  # 25 MB + C -> MB
    bfc.add_fusion(7, 2, 7)  # 26 MC + A -> MC

    # nonsense reactions, mRNA eats protein  cyclic backward
    bfc.add_fusion(1, 8, 1)  # 27 MA + C -> MA
    bfc.add_fusion(4, 2, 4)  # 28 MB + A -> MB
    bfc.add_fusion(7, 4, 7)  # 29 MC + B -> MC

    # nonsense reactions, protein eats protein self
    bfc.add_fusion(2, 2, 2)  # 30 A + A -> A
    bfc.add_fusion(5, 5, 5)  # 31 B + B -> B
    bfc.add_fusion(8, 8, 8)  # 32 C + C -> C

    # nonsense reactions, protein eats protein cyclic forward
    bfc.add_fusion(5, 2, 2)  # 30 B + A -> A
    bfc.add_fusion(8, 5, 5)  # 31 C + B -> B
    bfc.add_fusion(2, 8, 8)  # 32 A + C -> C

    # nonsense reactions, protein eats protein cyclic backward
    bfc.add_fusion(8, 2, 2)  # 33 C + A -> A
    bfc.add_fusion(5, 8, 8)  # 34 B + C -> C
    bfc.add_fusion(2, 5, 5)  # 35 A + B -> B

    # nonsense reactions, protein becomes protein cyclic forward
    bfc.add_conversion(2, 5)  # 36 A -> B
    bfc.add_conversion(5, 8)  # 37 B -> C
    bfc.add_conversion(8, 2)  # 38 C -> A

    # nonsense reactions, protein becomes protein cyclic backward
    bfc.add_conversion(2, 8)  # 39 A -> C
    bfc.add_conversion(8, 5)  # 40 C -> B
    bfc.add_conversion(5, 2)  # 41 B -> A

    # random reactions
    get_additional_funs(bfc)
    return bfc


def get_additional_funs(bfc):
    # species DA  MA  A  DB  MB  B  DC  MC  C
    # ids     0   1   2  3   4   5  6   7   8
    bfc.add_fusion(4, 7, 1)  # MB + MC -> MA, ok (causes lsq trouble)
    bfc.add_fusion(2, 7, 8)  # A + MC -> C, ok (causes lsq trouble)


def print_config(print_reactions=True):
    print("ReaDDy learn config")
    print("***" * 15)
    print("Target time: {}".format(TARGET_TIME))
    print("Gillespie intermediate rescaling factor: {}".format(SCALE))
    print("Discretization timestep: {}".format(TIMESTEP))
    if print_reactions:
        print("Reactions:")
        bfc = get_bfc_custom()
        assert len(DESIRED_RATES) == len(bfc.functions)
        data = []
        for rate, fun in zip(DESIRED_RATES, bfc.functions):
            if isinstance(fun, basis.ConversionReaction):
                data.append(["Conversion", "{} -> {}".format(SPECIES[fun.type1], SPECIES[fun.type2]), rate])
            elif isinstance(fun, basis.DecayReaction):
                data.append(["Decay", "{} -> 0".format(SPECIES[fun.type_from]), rate])
            elif isinstance(fun, basis.FissionReaction):
                data.append(["Fission", "{} -> {} + {}".format(SPECIES[fun.type_from], SPECIES[fun.type_to1],
                                                               SPECIES[fun.type_to2]), rate])
            elif isinstance(fun, basis.FusionReaction):
                data.append(["Fission", "{} + {} -> {}".format(SPECIES[fun.type_from1], SPECIES[fun.type_from2],
                                                               SPECIES[fun.type_to]), rate])
            elif isinstance(fun, basis.Intercept):
                data.append(["Intercept", "", rate])
            else:
                raise ValueError("unknown reaction function {}".format(fun))
        print(tabulate(data, headers=["Type", "Involved species", "Rate"]))


def get_regulation_network_lma(init=1):
    regulation_network = RegulationNetwork()
    regulation_network.timestep = TIMESTEP
    regulation_network.realisations = 1.
    regulation_network.noise_variance = 0.
    regulation_network.get_bfc = get_bfc_custom
    regulation_network.desired_rates = DESIRED_RATES
    regulation_network.target_time = TARGET_TIME
    print(len(regulation_network.desired_rates))
    print(regulation_network.get_bfc().n_basis_functions)
    regulation_network.initial_states = [regulation_network.initial_states[init]]
    analysis = regulation_network.generate_analysis_object(fname_prefix='case_1', fname_postfix='0')
    for i in range(len(regulation_network.initial_states)):
        analysis.generate_or_load_traj_lma(i, regulation_network.target_time,
                                           noise_variance=regulation_network.noise_variance,
                                           realizations=regulation_network.realisations)
        shape = analysis.trajs[i].counts.shape
        print("n_frames={}, n_species={}".format(*shape))
    regulation_network.compute_gradient_derivatives(analysis, persist=False)
    return regulation_network, analysis


def get_regulation_network_gillespie(gillespie_realisations=1):
    print("obtaining regulation network with dt = {}".format(TIMESTEP))
    regulation_network = RegulationNetwork()
    regulation_network.timestep = TIMESTEP
    regulation_network.realisations = 1.
    regulation_network.noise_variance = 0
    regulation_network.get_bfc = get_bfc_custom
    regulation_network.desired_rates = DESIRED_RATES
    regulation_network.target_time = TARGET_TIME
    # change init state here, case1 -> init1, case2 -> init1, case3 -> init1 and init3
    regulation_network.initial_states = [regulation_network.initial_states[3]]

    if SCALE > 1.:
        print("scaling population up, timestep down and bimol. rates down by factor {}".format(SCALE))
        regulation_network.desired_rates[18:21] /= SCALE
        regulation_network.initial_states = [
            [regulation_network.initial_states[0][i] * SCALE
             for i in range(len(regulation_network.initial_states[0]))]
        ]
        regulation_network.timestep /= SCALE

    analysis = regulation_network.generate_analysis_object(fname_prefix='case_2', fname_postfix='0')
    if gillespie_realisations is not None:
        print("generating data using gillespie kmc averaged over {} realisations".format(gillespie_realisations))
        for i in range(len(regulation_network.initial_states)):
            analysis.generate_or_load_traj_gillespie(i, target_time=TARGET_TIME,
                                                     n_realizations=gillespie_realisations,
                                                     update_and_persist=False, njobs=8)
    else:
        print("generating data by integrating the law of mass action with additive lma_noise {}".format(
            regulation_network.noise_variance))
        for i in range(len(regulation_network.initial_states)):
            analysis.generate_or_load_traj_lma(i, regulation_network.target_time,
                                               noise_variance=regulation_network.noise_variance,
                                               realizations=regulation_network.realisations)

    # before calculating derivatives scale population down again, timestep and bimolecular reaction rates up again
    if SCALE > 1.:
        for traj in analysis.trajs:
            traj.counts = traj.counts[::int(SCALE)] / SCALE
            traj.time_step = regulation_network.timestep * SCALE
            traj.update()
        regulation_network.desired_rates[18:21] *= SCALE
        regulation_network.timestep *= SCALE
        regulation_network.initial_states = [[regulation_network.initial_states[0][i] / SCALE
                                              for i in range(len(regulation_network.initial_states[0]))]]
    regulation_network.interpolate_counts(analysis, persist=False)
    regulation_network.compute_gradient_derivatives(analysis, persist=False)
    print(regulation_network.timestep)
    analysis2 = regulation_network.generate_analysis_object(fname_prefix='case_2_', fname_postfix='0')
    print(analysis2.timestep)
    analysis2._trajs = analysis.trajs
    for i in range(len(regulation_network.initial_states)):
        traj = analysis.get_traj(i)

    return regulation_network, analysis2


def get_traj_from_file(filename, n_gillespie_realisations, iid):
    with h5.File(filename, "r") as f:
        counts = f[str(n_gillespie_realisations)][str(iid)]["counts"][:]
        dcounts_dt = f[str(n_gillespie_realisations)][str(iid)]["dcounts_dt"][:]
    return counts, dcounts_dt


def integrate(initial_condition, rates):
    from scipy.integrate import odeint
    bfc = get_bfc_custom()

    def fun(data, _):
        theta = np.array([f(data) for f in bfc.functions])
        return np.matmul(rates, theta)

    xs = np.arange(0, TARGET_TIME, TIMESTEP)
    num_solution = odeint(fun, np.array(initial_condition).squeeze(), xs)

    return xs, num_solution
