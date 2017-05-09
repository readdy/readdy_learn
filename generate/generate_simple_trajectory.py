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
import numpy as np
import readdy._internal.readdybinding.common.io as io
from contextlib import closing
from readdy._internal.readdybinding.api import KernelProvider, Simulation
from readdy._internal.readdybinding.common import Vec
from readdy.util import platform_utils


def generate(n_timesteps, fname):
    kernel_provider = KernelProvider.get()
    kernel_provider.load_from_dir(platform_utils.get_readdy_plugin_dir())

    box_x, box_y, box_z = 10., 10., 10.

    time_step = .1

    sim = Simulation()
    sim.set_kernel("CPU")

    sim.kbt = 1.0
    sim.box_size = Vec(box_x, box_y, box_z)
    sim.periodic_boundary = [True, True, True]

    # diffusion constant = 1.0, radius = .0 (deprecated anyways)
    sim.register_particle_type("A", 1.0, 0.)
    sim.register_particle_type("B", 1.0, 0.)
    sim.register_particle_type("C", 1.0, 0.)

    # rate = .01, radius = .2
    sim.register_reaction_fusion("A+B->C", "A", "B", "C", .01, .2)

    n_a_particles = 200
    n_b_particles = 200

    a_particles_coordinates_x = np.random.uniform(0., box_x, n_a_particles) - .5 * box_x
    a_particles_coordinates_y = np.random.uniform(0., box_y, n_a_particles) - .5 * box_y
    a_particles_coordinates_z = np.random.uniform(0., box_z, n_a_particles) - .5 * box_z
    for x, y, z in zip(a_particles_coordinates_x, a_particles_coordinates_y, a_particles_coordinates_z):
        sim.add_particle("A", Vec(x, y, z))

    b_particles_coordinates_x = np.random.uniform(0, box_x, n_b_particles) - .5 * box_x
    b_particles_coordinates_y = np.random.uniform(0, box_y, n_b_particles) - .5 * box_y
    b_particles_coordinates_z = np.random.uniform(0, box_z, n_b_particles) - .5 * box_z
    for x, y, z in zip(b_particles_coordinates_x, b_particles_coordinates_y, b_particles_coordinates_z):
        sim.add_particle("B", Vec(x, y, z))

    # stride = 1
    handle = sim.register_observable_trajectory(1)

    with closing(io.File(fname, io.FileAction.CREATE, io.FileFlag.OVERWRITE)) as f:
        handle.enable_write_to_file(f, u"", int(3))
        sim.run_scheme_readdy(True).configure(time_step).run(n_timesteps)


if __name__ == '__main__':
    generate(2000, "simple_trajectory.h5")
