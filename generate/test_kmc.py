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
import numpy as np

import kinetic_monte_carlo as kmc

__license__ = "LGPL"
__author__ = "chrisfroe"


def example_system_conversions():
    """Produce realisations of system with only conversions"""
    n_species = 2
    n_boxes = 2
    diffusivity_0 = np.array([[0., 0.3], [0.4, 0.]])  # species 0
    diffusivity_1 = np.array([[0., 0.5], [0.9, 0.]])  # species 1
    diffusivity = np.array([diffusivity_0, diffusivity_1])
    init_state = np.array([[1, 1], [2, 2]], dtype=np.int)
    system = kmc.ReactionDiffusionSystem(diffusivity, n_species, n_boxes, init_state)
    system.add_conversion(0, 1, np.array([4., 4.]))
    system.add_conversion(1, 0, np.array([0.5, 0.5]))
    system.simulate(50)
    return system


class TestKineticMonteCarlo(unittest.TestCase):
    def test_raise_if_finalized(self):
        with np.testing.assert_raises(RuntimeError):
            n_species, n_boxes = 2, 2
            diffusivity_0 = np.array([[0., 0.3], [0.4, 0.]])  # species 0
            diffusivity_1 = np.array([[0., 0.5], [0.9, 0.]])  # species 1
            diffusivity = np.array([diffusivity_0, diffusivity_1])
            init_state = np.array([[1, 1], [2, 2]], dtype=np.int)
            system = kmc.ReactionDiffusionSystem(diffusivity, n_species, n_boxes, init_state)
            system.simulate(5)
            system.add_creation(1, 1.)

    def test_always_positive_number_of_particles(self):
        event_list, time_list, state_list = example_system_conversions().sequence
        state_array = np.asarray(state_list, dtype=np.int)
        all_positive = state_array >= 0
        np.testing.assert_equal(np.all(all_positive), True)

    def test_conservation_of_particles(self):
        """In the system of only Conversion reactions, the total number of particles is conserved."""
        event_list, time_list, state_list = example_system_conversions().sequence
        n_particles = np.sum(state_list[0])
        all_correct = np.fromiter(map(lambda state: np.sum(state) == n_particles, state_list), dtype=np.bool)
        np.testing.assert_equal(np.all(all_correct), True)

    def test_convert_to_time_series_args(self):
        with np.testing.assert_raises(Exception):
            system = example_system_conversions()
            system.convert_events_to_time_series([])

        with np.testing.assert_raises(Exception):
            system = example_system_conversions()
            system.convert_events_to_time_series([], time_step=0.1, n_frames=10)

    def test_conservation_of_particles_after_converting(self):
        system = example_system_conversions()
        time_series, times = system.convert_events_to_time_series(n_frames=500)
        n_particles = np.sum(time_series[0])
        all_correct = np.fromiter(map(lambda state: np.sum(state) == n_particles, time_series), dtype=np.bool)
        np.testing.assert_equal(np.all(all_correct), True)
