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

"""Produce realisations of the reaction-diffusion-master-equation

This algorithm is known as: kinetic Monte Carlo, stochastic simulation algorithm, Gillespie algorithm.
"""

__license__ = "LGPL"
__author__ = "chrisfroe"


class Event:
    pass


class ReactionEvent(Event):
    pass


class DiffusionEvent(Event):
    pass


class Reaction:
    def __init__(self):
        self.rate_constant = 0
        self.propensity_fun = lambda x: 0


class ReactionDiffusionSystem:
    def __init__(self, diffusivity, reactions, n_species, n_boxes):
        # @todo check dimensions
        self._event_list = []
        pass

    def set_state(self, state):
        pass

    def simulate(self, n_steps):
        pass

    @property
    def event_list(self):
        return self._event_list


def generate_event_sequence(n_steps, init_state, diffusivity, reactions, n_species, n_boxes):
    system = ReactionDiffusionSystem(diffusivity, reactions, n_species, n_boxes)
    system.set_state(init_state)
    system.simulate(n_steps)
    return system.event_list
