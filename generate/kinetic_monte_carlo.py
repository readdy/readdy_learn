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

import numpy as np
import bisect
import math

__license__ = "LGPL"
__author__ = "chrisfroe"


class Event:
    """Event is performed _on_ the system and applies a delta.

    The delta can be across boxes or across species, which is why its dimensions can be either (n_boxes,) or (n_species,).
    This behavior is defined by the derived classes.
    """

    def __init__(self, stoichiometric_delta, cumulative_propensity):
        self._stoichiometric_delta = stoichiometric_delta
        self.cumulative_propensity = cumulative_propensity

    def perform(self, state):
        pass


class ReactionEvent(Event):
    def __init__(self, box_idx, stoichiometric_delta, cumulative_propensity):
        super(ReactionEvent, self).__init__(stoichiometric_delta, cumulative_propensity)
        self._box_idx = box_idx

    def perform(self, state):
        state[self._box_idx] += self._stoichiometric_delta


class DiffusionEvent(Event):
    def __init__(self, species_idx, stoichiometric_delta, cumulative_propensity):
        super(DiffusionEvent, self).__init__(stoichiometric_delta, cumulative_propensity)
        self._species_idx = species_idx

    def perform(self, state):
        state[:, self._species_idx] += self._stoichiometric_delta


class Reaction:
    """Reactions behave like structs, that carry the rates and stoichiometric information for all possible reactions"""

    def __init__(self, rate, n_species):
        self.rate = rate
        self.stoichiometric_delta = np.zeros(n_species, dtype=np.int)


class Conversion(Reaction):
    def __init__(self, species_from, species_to, rate, n_species):
        super(Conversion, self).__init__(rate, n_species)
        self.species_from = species_from
        self.species_to = species_to
        self.stoichiometric_delta[self.species_from] = -1
        self.stoichiometric_delta[self.species_to] = +1

    def propensity(self, box_state):
        return box_state[self.species_from]


class ReactionDiffusionSystem:
    def __init__(self, diffusivity, reactions, n_species, n_boxes, init_state, init_time=0.):
        assert n_species > 0
        assert n_boxes > 0
        # diffusivity can be a list of sparse matrices or a rank 3 tensor
        assert len(diffusivity) == n_species
        assert diffusivity[0].shape == (n_boxes, n_boxes,)
        assert init_state.shape == (n_boxes, n_species)
        self._n_species = n_species
        self._n_boxes = n_boxes
        self._n_reactions = len(reactions)
        self._diffusivity = diffusivity
        self._reactions = reactions
        self._state = np.copy(init_state)
        self._init_state = np.copy(init_state)
        self._time = init_time
        self._event_list = []
        self._times_list = [init_time]
        self._state_list = [init_state]

    def simulate(self, n_steps):
        for t in range(n_steps):
            possible_events = []
            cumulative = 0.
            # gather reaction events
            for i in range(self._n_boxes):
                for r in range(self._n_reactions):
                    propensity = self._reactions[r].propensity(self._state[i])
                    cumulative += propensity
                    delta = self._reactions[r].stoichiometric_delta
                    possible_events.append(ReactionEvent(i, delta, cumulative))
            # gather diffusion events
            for s in range(self._n_species):
                for i in range(self._n_boxes):
                    for j in range(self._n_boxes):
                        propensity = self._diffusivity[s][i, j] * self._state[i, s]
                        cumulative += propensity
                        delta = np.zeros(self._n_boxes, dtype=np.int)
                        delta[i] = -1
                        delta[j] = +1
                        possible_events.append(DiffusionEvent(s, delta, cumulative))
            # draw time and cumulative value
            event_time = (1. / cumulative) * np.log(1. / np.random.random())
            rnd = np.random.random() * cumulative
            # find event corresponding to rnd, that shall be performed
            cumulative_list = [x.cumulative_propensity for x in possible_events]
            event_idx = bisect.bisect_right(cumulative_list, rnd) - 1
            event = possible_events[event_idx]
            # save event and time to sequence
            self._event_list.append(event)
            self._times_list.append(self._time + event_time)
            # update system and save state
            self._time += event_time
            event.perform(self._state)
            self._state_list.append(self._state)

    @property
    def sequence(self):
        return self._event_list, self._times_list, self._state_list

    # @todo move this into a result(event sequence)-object
    def convert_to_time_series(self, time_step=None, n_frames=None):
        if not ((time_step is not None) ^ (n_frames is not None)):
            raise RuntimeError("Either time_step (x)or n_frames must be given")
        if len(self._times_list) < 2:
            raise RuntimeError("Sample some events first")
        if n_frames:
            time_step = (self._times_list[-1] - self._times_list[0]) / float(n_frames)
        else:
            n_frames = math.ceil((self._times_list[-1] - self._times_list[0]) / time_step)

        result = np.zeros((n_frames, self._n_boxes, self._n_species), dtype=np.int)
        current_t = self._times_list[0]
        last_passed_event_time_idx = 0
        print("\nresult.shape\n", result.shape)
        print("\nself._init_state\n", self._init_state)
        result[0,:,:] = self._init_state
        for t in range(1, n_frames):
            current_t += time_step
            n_passed_events = 0
            if current_t > self._times_list[last_passed_event_time_idx + 1]:
                n_passed_events = 1
                # if we have skipped one event we might have skipped another
                while current_t > self._times_list[last_passed_event_time_idx + n_passed_events + 1]:
                    n_passed_events += 1
            last_passed_event_time_idx += n_passed_events
            result[t] = self._state_list[last_passed_event_time_idx]
        return result


def generate_event_sequence(n_steps, init_state, diffusivity, reactions, n_species, n_boxes):
    system = ReactionDiffusionSystem(diffusivity, reactions, n_species, n_boxes, init_state)
    system.simulate(n_steps)
    return system.sequence
