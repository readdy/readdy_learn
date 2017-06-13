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

import unittest
import tempfile
import shutil
import numpy as np

import kinetic_monte_carlo as kmc

__license__ = "LGPL"
__author__ = "chrisfroe"


def example_system():
    n_species = 2
    n_boxes = 2
    diffusivity_0 = np.array([[0., 0.3], [0.4, 0.]])  # species 0
    diffusivity_1 = np.array([[0., 0.5], [0.9, 0.]])  # species 1
    diffusivity = np.array([diffusivity_0, diffusivity_1])
    reactions = [kmc.Conversion(0, 1, 4., n_species), kmc.Conversion(1, 0, 0.5, n_species)]
    print("reactions[1].stoichiometric_delta", reactions[1].stoichiometric_delta)
    init_state = np.array([[1, 1], [2, 2]], dtype=np.int)
    system = kmc.ReactionDiffusionSystem(diffusivity, reactions, n_species, n_boxes, init_state)
    system.simulate(50)
    return system


class TestKineticMonteCarlo(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #    cls.dir = tempfile.mkdtemp("test-kmc")

    # @classmethod
    # def tearDownClass(cls):
    #    shutil.rmtree(cls.dir, ignore_errors=True)

    def test_sanity(self):
        print("\nEreignisse", example_system().sequence)

    def test_conversion_args(self):
        with np.testing.assert_raises(Exception):
            system = example_system()
            system.convert_to_time_series([])

        with np.testing.assert_raises(Exception):
            system = example_system()
            system.convert_to_time_series([], time_step=0.1, n_frames=10)

    def test_conversion_sanity(self):
        system = example_system()

        time_series = system.convert_to_time_series(n_frames=500)
        print("time_series", time_series)