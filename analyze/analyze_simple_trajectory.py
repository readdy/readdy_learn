# coding=utf-8

# Copyright © 2016 Computational Molecular Biology Group,
#                  Freie Universität Berlin (GER)
#
# This file is part of ReaDDy.
#
# ReaDDy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General
# Public License along with this program. If not, see
# <http://www.gnu.org/licenses/>.

"""
Created on 10.05.17

@author: clonker
"""
import os

import analyze.log as log
import analyze.trajectory_reader as tr
import analyze_tools.analyze_tools as at
import h5py


def get_discrete_trajectory(fname):
    dtraj = []
    if os.path.exists(fname):
        with h5py.File(fname) as f:
            traj = f["readdy/trajectory"]
            traj_time = traj["time"]
            traj_time_records = traj["records"]
            for time, records in zip(traj_time, traj_time_records):
                current_counts = {}
                for record in records:
                    type_id = record["typeId"]
                    if type_id in current_counts.keys():
                        current_counts[type_id] += 1
                    else:
                        current_counts[type_id] = 1
                dtraj.append(current_counts)
    else:
        log.warn("file {} did not exist".format(fname))

    return dtraj

if __name__ == '__main__':
    log.set_level("debug")
    dtraj = get_discrete_trajectory("../generate/simple_trajectory.h5")
    type_mapping = {"A": 0, "B": 1, "C": 2}
    for i in range(100):
        print(dtraj[i])
