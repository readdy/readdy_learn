import numpy as _np
import pynumtools.kmc as kmc
import readdy_learn.analyze.basis as basis
import matplotlib.pyplot as plt


class RegulationNetwork(object):

    def __init__(self):
        # species DA  MA  A  DB  MB  B  DC  MC  C
        # ids     0   1   2  3   4   5  6   7   8
        self.n_species = 9
        self.species_names = ["DA", "MA", "A", "DB", "MB", "B", "DC", "MC", "C"]
        self.desired_rates = _np.array([
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

        initial_states = [
            [1, 0, 0, 1, 0, 0, 1, 0, 0], [1, 2, 0, 1, 0, 3, 1, 0, 0], [1, 1, 2, 1, 0, 2.5, 1, 0, 2],
            [1, 1, 2, 1, 0, 0, 1, 3, 0]
        ]
        self.initial_states = [_np.array([arr]) for arr in initial_states]

        self.ld_derivative_config = {
            'ld_derivative_atol': 1e-4,
            'ld_derivative_rtol': None,
            'ld_derivative_linalg_solver_maxit': 100000,
            'ld_derivative_alpha': 1e-1,
            'ld_derivative_solver': 'spsolve',
            'ld_derivative_linalg_solver_tol': 1e-10,
            'ld_derivative_use_preconditioner': False
        }

        self.noise_variance = 0.0001
        self.target_time = 3.
        self.realisations = 60
        self.timestep = 1e-3

    def set_up_system(self, init_state):
        sys = kmc.ReactionDiffusionSystem(diffusivity=self.n_species * [[[0.]]], n_species=self.n_species, n_boxes=1,
                                          init_state=init_state, species_names=self.species_names)
        # usual stuff A
        sys.add_fission("DA", "DA", "MA", _np.array([self.desired_rates[0]]))  # DA -> DA + MA transcription
        sys.add_fission("MA", "MA", "A", _np.array([self.desired_rates[1]]))  # MA -> MA + A translation
        sys.add_decay("MA", _np.array([self.desired_rates[2]]))  # MA -> 0 mRNA A decay
        sys.add_decay("A", _np.array([self.desired_rates[3]]))  # A -> 0 protein decay
        # usual stuff B
        sys.add_fission("DB", "DB", "MB", _np.array([self.desired_rates[4]]))  # DB -> DB + MB transcription
        sys.add_fission("MB", "MB", "B", _np.array([self.desired_rates[5]]))  # MB -> MB + B translation
        sys.add_decay("MB", _np.array([self.desired_rates[6]]))  # MB -> 0 mRNA B decay
        sys.add_decay("B", _np.array([self.desired_rates[7]]))  # B -> 0 protein decay
        # usual stuff C
        sys.add_fission("DC", "DC", "MC", _np.array([self.desired_rates[8]]))  # DC -> DC + MC transcription
        sys.add_fission("MC", "MC", "C", _np.array([self.desired_rates[9]]))  # MC -> MC + C translation
        sys.add_decay("MC", _np.array([self.desired_rates[10]]))  # MC -> 0 mRNA C decay
        sys.add_decay("C", _np.array([self.desired_rates[11]]))  # C -> 0 protein decay

        # regulation: only the real ones show up here
        # self regulation
        # sys.add_fusion("MA", "A", "A", _np.array([desired_rates[12]]))  # MA + A -> A, A regulates A
        # sys.add_fusion("MB", "B", "B", _np.array([desired_rates[13]]))  # MB + B -> B, B regulates B
        # sys.add_fusion("MC", "C", "C", _np.array([desired_rates[14]]))  # MC + C -> C, C regulates C
        # cyclic forward
        # sys.add_fusion("MB", "A", "A", _np.array([desired_rates[15]])) # MB + A -> A, A regulates B
        # sys.add_fusion("MC", "B", "B", _np.array([desired_rates[16]])) # MC + B -> B, B regulates C
        # sys.add_fusion("MA", "C", "C", _np.array([desired_rates[17]])) # MA + C -> C, C regulates A
        # cyclic backward
        sys.add_fusion("MC", "A", "A", _np.array([self.desired_rates[18]]))  # MC + A -> A, A regulates C
        sys.add_fusion("MB", "C", "C", _np.array([self.desired_rates[19]]))  # MB + C -> C, C regulates B
        sys.add_fusion("MA", "B", "B", _np.array([self.desired_rates[20]]))  # MA + B -> B, B regulates A

        return sys

    def get_bfc(self):
        # species DA  MA  A  DB  MB  B  DC  MC  C
        # ids     0   1   2  3   4   5  6   7   8
        bfc = basis.BasisFunctionConfiguration(self.n_species)
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

        assert bfc.n_basis_functions == len(self.desired_rates), \
            "got {} basis functions but only {} desired rates".format(bfc.n_basis_functions, len(self.desired_rates))
        return bfc

    def plot_concentrations(self, analysis, traj_n):
        traj_number = traj_n
        pfs = [
            lambda t=traj_number: analysis.plot_derivatives(t, species=[0]),
            lambda t=traj_number: analysis.plot_concentration_curves(t, species=[0]),
            lambda t=traj_number: analysis.plot_derivatives(t, species=[3]),
            lambda t=traj_number: analysis.plot_concentration_curves(t, species=[3]),
            lambda t=traj_number: analysis.plot_derivatives(t, species=[6]),
            lambda t=traj_number: analysis.plot_concentration_curves(t, species=[6]),
            lambda t=traj_number: analysis.plot_derivatives(t, species=[1, 2]),
            lambda t=traj_number: analysis.plot_concentration_curves(t, species=[1, 2]),
            lambda t=traj_number: analysis.plot_derivatives(t, species=[4, 5]),
            lambda t=traj_number: analysis.plot_concentration_curves(t, species=[4, 5]),
            lambda t=traj_number: analysis.plot_derivatives(t, species=[7, 8]),
            lambda t=traj_number: analysis.plot_concentration_curves(t, species=[7, 8])
        ]
        analysis.plot_overview(pfs, n_cols=2, n_rows=6, size_factor=1.3)
        plt.show()

    def generate_analysis_object(self, fname_prefix="regulation_network", fname_postfix=""):
        import readdy_learn.analyze.analyze as ana
        analysis = ana.ReactionAnalysis(self.get_bfc(), self.desired_rates, self.initial_states, self.set_up_system,
                                        fname_prefix=fname_prefix, fname_postfix=fname_postfix,
                                        n_species=self.n_species, timestep=self.timestep,
                                        ld_derivative_config=self.ld_derivative_config, recompute_traj=False,
                                        species_names=self.species_names)
        return analysis

    def compute_gradient_derivatives(self, analysis, persist=True):
        for t in range(len(self.initial_states)):
            traj = analysis.get_traj(t)
            # for sp in [0, 3, 6]:
            #    dx = _np.zeros_like(traj.counts[:, sp])
            #    print("species {} dx.shape {}".format(sp, dx.shape))
            #    traj.separate_derivs[sp] = dx
            for sp in [1, 2, 4, 5, 7, 8, 0, 3, 6]:
                x = traj.counts[:, sp]
                dt = traj.time_step
                dx = _np.gradient(x) / dt
                traj.separate_derivs[sp] = dx
            if persist:
                traj.persist()
