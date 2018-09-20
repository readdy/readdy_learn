#!/usr/bin/env python
import argparse
import numpy as np
import samplebase as sb
import h5py as h5
import time

import readdy_learn.analyze.basis as basis
import readdy_learn.analyze.cross_validation as cross_validation
import readdy_learn.analyze.tools as tools
from readdy_learn.example.regulation_network import RegulationNetwork
from readdy_learn.example.regulation_network import DEFAULT_DESIRED_RATES

"""Usage: `python run_cv.py prefix name`"""

parser = argparse.ArgumentParser(description='Run simulation and output data to the given workspace prefix.')
parser.add_argument('prefix', type=str, help='prefix of sample, directory that contains the sample directory')
parser.add_argument('name', type=str, help='name of sample')


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


def get_n_additional_funs():
    return 2


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

desired_rates = np.append(DESIRED_RATES, np.zeros((get_n_additional_funs(),)))


def get_regulation_network(timestep, lma_noise=0., target_time=2., gillespie_realisations=1, scale=1.):
    assert np.floor(scale) == scale
    print("obtaining regulation network with dt = {}".format(timestep))
    regulation_network = RegulationNetwork()
    regulation_network.timestep = timestep
    regulation_network.realisations = 1.
    regulation_network.noise_variance = lma_noise
    regulation_network.get_bfc = get_bfc_custom
    regulation_network.desired_rates = desired_rates
    regulation_network.target_time = target_time
    # change init state here, case1 -> init1, case2 -> init1, case3 -> init1 and init2
    regulation_network.initial_states = [regulation_network.initial_states[3]]

    if scale > 1.:
        print("scaling population up, timestep down and bimol. rates down by factor {}".format(scale))
        regulation_network.desired_rates[18:21] /= scale
        regulation_network.initial_states = [
            [regulation_network.initial_states[0][i] * scale
             for i in range(len(regulation_network.initial_states[0]))]
        ]
        regulation_network.timestep /= scale

    analysis = regulation_network.generate_analysis_object(fname_prefix='case_2', fname_postfix='0')
    if gillespie_realisations is not None:
        print("generating data using gillespie kmc averaged over {} realisations".format(gillespie_realisations))
        for i in range(len(regulation_network.initial_states)):
            analysis.generate_or_load_traj_gillespie(i, target_time=target_time,
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
    if scale > 1.:
        for traj in analysis.trajs:
            traj.counts = traj.counts[::int(scale)] / scale
            traj.time_step = regulation_network.timestep * scale
            traj.update()
        regulation_network.desired_rates[18:21] *= scale
        regulation_network.timestep *= scale
        regulation_network.initial_states = [[regulation_network.initial_states[0][i] / scale
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


def generate_counts(dt=3e-3, lma_noise=0., target_time=2., gillespie_realisations=1):
    print(
        "run generate_traj with splitter='kfold', dt={}, lma_noise={}, target_time={}, gillespie_realisations={}".format(
            dt, lma_noise, target_time, gillespie_realisations))
    regulation_network, analysis = get_regulation_network(dt, lma_noise=lma_noise, target_time=target_time,
                                                          gillespie_realisations=gillespie_realisations, scale=500)
    traj = analysis.get_traj(0)
    return traj.counts, traj.dcounts_dt, traj.time_step


def create_traj_file(traj_file_path="./gillespie_trajs_init_3.h5", dt=3e-3, target_time=2.,
                     realisations=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                     number_of_iids=10):
    with h5.File(traj_file_path, "w") as f:
        for r in realisations:
            r_group = f.create_group(str(r))
            for i in range(number_of_iids):
                i_group = r_group.create_group(str(i))

                counts, dcounts_dt, timestep = generate_counts(
                    dt=dt, lma_noise=0., target_time=target_time, gillespie_realisations=r)

                counts_dset = i_group.create_dataset("counts", data=counts)
                counts_dset.attrs["timestep"] = dt
                i_group.create_dataset("dcounts_dt", data=dcounts_dt)


def concatenate_two_traj_files(traj1="./gillespie_trajs_init_1.h5", traj2="./gillespie_trajs_init_3.h5",
                               traj_out="./gillespie_trajs_conced_1_3_normal.h5",
                               realisations=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                               number_of_iids=10, zipped=False):
    with h5.File(traj_out, "w") as f_out:
        with h5.File(traj1, "r") as f1:
            with h5.File(traj2, "r") as f2:
                for r in realisations:
                    r_out_group = f_out.create_group(str(r))
                    for i in range(number_of_iids):
                        i_out_group = r_out_group.create_group(str(i))

                        c1 = f1[str(r)][str(i)]["counts"][:]
                        c2 = f2[str(r)][str(i)]["counts"][:]

                        dt1 = f1[str(r)][str(i)]["counts"].attrs["timestep"]
                        dt2 = f2[str(r)][str(i)]["counts"].attrs["timestep"]

                        assert dt1 == dt2

                        dc_dt1 = f1[str(r)][str(i)]["dcounts_dt"][:]
                        dc_dt2 = f2[str(r)][str(i)]["dcounts_dt"][:]

                        print("c1.shape", c1.shape)

                        if zipped:
                            # @todo zip
                            counts = np.stack((c1, c2))
                            dcounts_dt = np.stack((dc_dt1, dc_dt2))
                            print("counts.shape", counts.shape)
                            new_shape = counts.shape[1:]
                            new_shape = (new_shape[0] * 2, *(new_shape[1:]))
                            print("new_shape", new_shape)
                            counts = np.reshape(counts, new_shape, order='F')
                            dcounts_dt = np.reshape(dcounts_dt, new_shape, order='F')
                        else:
                            counts = np.concatenate((c1, c2), axis=0)
                            dcounts_dt = np.concatenate((dc_dt1, dc_dt2), axis=0)

                        counts_dset = i_out_group.create_dataset("counts", data=counts)
                        counts_dset.attrs["timestep"] = dt1
                        i_out_group.create_dataset("dcounts_dt", data=dcounts_dt)


# do this for "./gillespie_trajs_init_1.h5" and "./gillespie_trajs_conced_1_3_zipped.h5"
def shuffle_trajs(traj_file_path="./gillespie_trajs_init_1.h5",
                     realisations=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                     number_of_iids=10):
    with h5.File(traj_file_path, "r+") as f:
        for r in realisations:
            #r_group = f.create_group(str(r))
            for i in range(number_of_iids):
                #i_group = r_group.create_group(str(i))

                c1 = f[str(r)][str(i)]["counts"][:]
                dc_dt1 = f[str(r)][str(i)]["dcounts_dt"][:]

                h1 = f[str(r)][str(i)]["counts"]
                h2 = f[str(r)][str(i)]["dcounts_dt"]

                indices = np.arange(len(c1))
                np.random.shuffle(indices)

                h1[:] = c1[indices]
                h2[:] = dc_dt1[indices]


def get_traj_from_file(filename, n_gillespie_realisations, iid):
    with h5.File(filename, "r") as f:
        counts = f[str(n_gillespie_realisations)][str(iid)]["counts"][:]
        dcounts_dt = f[str(n_gillespie_realisations)][str(iid)]["dcounts_dt"][:]
    return counts, dcounts_dt

# gillespie_realisations = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
# n_iids = 10
def do_cv(alpha=1., n_splits=5, gillespie_realisations=1, iid_id=0, traj_file_path="./gillespie_two_inits_conced.h5"):
    with h5.File(traj_file_path, "r") as f:
        counts_dset = f[str(gillespie_realisations)][str(iid_id)]["counts"]
        counts = counts_dset[:]
        timestep = counts_dset.attrs["timestep"]
        dcounts_dt = f[str(gillespie_realisations)][str(iid_id)]["dcounts_dt"][:]
    traj = tools.Trajectory(counts, time_step=timestep)
    traj.dcounts_dt = dcounts_dt
    cv = cross_validation.CrossValidation([traj], get_bfc_custom())
    cv.splitter = 'kfold'
    cv.n_splits = n_splits
    cv.njobs = 1
    cv.show_progress = True
    l1_ratios = np.array([1.])  # np.linspace(0, 1, num=5)
    cv_result = cv.cross_validate(alpha, l1_ratios, realizations=1)
    # rates = analysis.solve(0, alpha, l1_ratio=1., tol=1e-16, recompute=True, persist=False, concatenated=True)
    result = {"cv_result": cv_result}
    return result


def estimate(alpha=1., gillespie_realisations=1, iid_id=0, traj_file_path="./gillespie_trajs.h5"):
    with h5.File(traj_file_path, "r") as f:
        counts_dset = f[str(gillespie_realisations)][str(iid_id)]["counts"]
        counts = counts_dset[:]
        timestep = counts_dset.attrs["timestep"]
        dcounts_dt = f[str(gillespie_realisations)][str(iid_id)]["dcounts_dt"][:]
    return estimate_counts(alpha=alpha, counts=counts, dcounts_dt=dcounts_dt, timestep=timestep)


def estimate_counts(alpha, counts, dcounts_dt, timestep):
    regulation_network, analysis = get_regulation_network(timestep, 0., 2., gillespie_realisations=1, scale=500.)
    traj = tools.Trajectory(counts, time_step=timestep)
    traj.dcounts_dt = dcounts_dt
    analysis._trajs = [traj]

    tolerances_to_try = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8]

    rates = None
    for tol in tolerances_to_try:
        try:
            print("Trying tolerance {}".format(tol))
            rates = analysis.solve(0, alpha=alpha, l1_ratio=1., tol=tol, recompute=True, persist=False,
                                   concatenated=True)
            break
        except ValueError:
            print("... this tolerance {} failed".format(tol))
            if tol == tolerances_to_try[len(tolerances_to_try) - 1]:
                raise

    return rates, analysis


if __name__ == '__main__':
    print("run_cv ..")
    t1 = time.perf_counter()

    args = parser.parse_args()

    print("realization prefix {}".format(args.prefix))

    with sb.SampleContextManager(args.prefix, args.name) as sample:
        print("using traj file {}".format(sample.args["traj_file_path"]))
        sample.result = do_cv(**sample.args)

    t2 = time.perf_counter()
    print("done run_cv in {} seconds".format(t2 - t1))
