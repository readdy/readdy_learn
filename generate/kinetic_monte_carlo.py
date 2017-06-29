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
A proper description of such an algorithm is given by

    R. Erban, J. Chapman, and P. Maini, "A practical guide to stochastic simulations of reaction-diffusion processes", pp. 24–29, Apr. 2007.
"""

import numpy as np
import bisect
import math
import copy
import logging

import logutil

__license__ = "LGPL"
__author__ = "chrisfroe"

logging.basicConfig(format='[KMC] [%(asctime)s] [%(levelname)s] %(message)s', level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
log = logutil.StyleAdapter(logging.getLogger(__name__))


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

    def __str__(self):
        string = "ReactionEvent\n"
        string += "box_idx " + str(self._box_idx) + "\n"
        string += "stoichiometric_delta (within one box, across species) " + str(self._stoichiometric_delta) + "\n"
        string += "cumulative_propensity " + str(self.cumulative_propensity)
        return string

    def perform(self, state):
        state[self._box_idx] += self._stoichiometric_delta


class DiffusionEvent(Event):
    def __init__(self, species_idx, stoichiometric_delta, cumulative_propensity):
        super(DiffusionEvent, self).__init__(stoichiometric_delta, cumulative_propensity)
        self._species_idx = species_idx

    def __str__(self):
        string = "DiffusionEvent\n"
        string += "species_idx " + str(self._species_idx) + "\n"
        string += "stoichiometric_delta (within one species, across boxes) " + str(self._stoichiometric_delta) + "\n"
        string += "cumulative_propensity " + str(self.cumulative_propensity)
        return string

    def perform(self, state):
        state[:, self._species_idx] += self._stoichiometric_delta


class Reaction:
    """Reactions behave like structs, that carry the rates and stoichiometric information for all possible reactions"""

    def __init__(self, rate, n_species, n_boxes):
        assert len(rate) == n_boxes
        self.rate = rate
        self.stoichiometric_delta = np.zeros(n_species, dtype=np.int)


class Conversion(Reaction):
    def __init__(self, species_from, species_to, rate, n_species, n_boxes):
        super(Conversion, self).__init__(rate, n_species, n_boxes)
        self.species_from = species_from
        self.species_to = species_to
        self.stoichiometric_delta[self.species_from] = -1
        self.stoichiometric_delta[self.species_to] = +1

    def propensity(self, box_state, box_idx):
        return self.rate[box_idx] * box_state[self.species_from]


class Fusion(Reaction):
    def __init__(self, species_from1, species_from2, species_to, rate, n_species, n_boxes):
        super(Fusion, self).__init__(rate, n_species, n_boxes)
        self.species_from1 = species_from1
        self.species_from2 = species_from2
        self.species_to = species_to
        self.stoichiometric_delta[self.species_from1] = -1
        self.stoichiometric_delta[self.species_from2] = -1
        self.stoichiometric_delta[self.species_to] = +1

    def propensity(self, box_state, box_idx):
        return self.rate[box_idx] * box_state[self.species_from1] * box_state[self.species_from2]


class Fission(Reaction):
    def __init__(self, species_from, species_to1, species_to2, rate, n_species, n_boxes):
        super(Fission, self).__init__(rate, n_species, n_boxes)
        self.species_from = species_from
        self.species_to1 = species_to1
        self.species_to2 = species_to2
        self.stoichiometric_delta[self.species_from] = -1
        self.stoichiometric_delta[self.species_to1] = +1
        self.stoichiometric_delta[self.species_to2] = +1

    def propensity(self, box_state, box_idx):
        return self.rate[box_idx] * box_state[self.species_from]


class Decay(Reaction):
    def __init__(self, species_from, rate, n_species, n_boxes):
        super(Decay, self).__init__(rate, n_species, n_boxes)
        self.species_from = species_from
        self.stoichiometric_delta[self.species_from] = -1

    def propensity(self, box_state, box_idx):
        return self.rate[box_idx] * box_state[self.species_from]


class Creation(Reaction):
    def __init__(self, species_to, rate, n_species, n_boxes):
        super(Creation, self).__init__(rate, n_species, n_boxes)
        self.species_to = species_to
        self.stoichiometric_delta[self.species_to] = +1

    def propensity(self, box_state, box_idx):
        return self.rate[box_idx]


class TrajectoryConfig(object):
    def __init__(self, types, reactions):
        self.types = types
        self.reactions = reactions


class ReactionDiffusionSystem:
    def __init__(self, diffusivity, n_species, n_boxes, init_state, init_time=0., species_names=None):
        assert n_species > 0
        assert n_boxes > 0
        # diffusivity can be a list of sparse matrices or a rank 3 tensor
        assert len(diffusivity) == n_species
        assert diffusivity[0].shape == (n_boxes, n_boxes,)
        assert init_state.shape == (n_boxes, n_species)
        self._n_species = n_species
        self._n_boxes = n_boxes
        self._diffusivity = diffusivity
        self._reactions = []
        self._n_reactions = 0
        self._state = np.copy(init_state)
        self._init_state = np.copy(init_state)
        self._time = init_time
        self._event_list = []
        self._time_list = [init_time]
        self._state_list = [init_state]
        self._is_finalized = False
        if species_names:
            assert len(species_names) == n_species
        else:
            species_names = [str(x) for x in range(n_species)]
        self._species_names = species_names

    def get_trajectory_config(self):
        """Generate a TrajectoryConfig object for interoperability with readdy_learn

        :return: TrajectoryConfig object
        """
        types = dict()
        for idx, name in enumerate(self._species_names):
            types[name] = idx
        traj_config = TrajectoryConfig(types, self._reactions)
        return traj_config

    def __str__(self):
        string = "ReactionDiffusionSystem\n"
        string += "--- n_species " + str(self._n_species) + "\n"
        string += "--- n_boxes " + str(self._n_boxes) + "\n"
        string += "--- diffusivity\n" + str(self._diffusivity) + "\n"
        string += "--- reactions\n" + str(self._reactions) + "\n"
        string += "--- state\n" + str(self._state) + "\n"
        # string += "--- init_state\n" + str(self._init_state) + "\n"
        string += "--- time " + str(self._time) + "\n"
        # string += "--- event_list\n" + str(self._event_list) + "\n"
        # string += "--- time_list\n" + str(self._time_list) + "\n"
        # string += "--- state_list\n" + str(self._state_list) + "\n"
        return string

    def _assure_not_finalized(self):
        if not self._is_finalized:
            return
        else:
            raise RuntimeError("System has already been finalized")

    def add_conversion(self, species_from, species_to, rate):
        self._assure_not_finalized()
        conversion = Conversion(species_from, species_to, rate, self._n_species, self._n_boxes)
        self._reactions.append(conversion)
        self._n_reactions = len(self._reactions)

    def add_fusion(self, species_from1, species_from2, species_to, rate):
        self._assure_not_finalized()
        fusion = Fusion(species_from1, species_from2, species_to, rate, self._n_species, self._n_boxes)
        self._reactions.append(fusion)
        self._n_reactions = len(self._reactions)

    def add_fission(self, species_from, species_to1, species_to2, rate):
        self._assure_not_finalized()
        fission = Fission(species_from, species_to1, species_to2, rate, self._n_species, self._n_boxes)
        self._reactions.append(fission)
        self._n_reactions = len(self._reactions)

    def add_decay(self, species_from, rate):
        self._assure_not_finalized()
        decay = Decay(species_from, rate, self._n_species, self._n_boxes)
        self._reactions.append(decay)
        self._n_reactions = len(self._reactions)

    def add_creation(self, species_to, rate):
        self._assure_not_finalized()
        creation = Creation(species_to, rate, self._n_species, self._n_boxes)
        self._reactions.append(creation)
        self._n_reactions = len(self._reactions)

    @property
    def n_species(self):
        return self._n_species

    @property
    def n_boxes(self):
        return self.n_boxes

    @property
    def n_reactions(self):
        return self._n_reactions

    @property
    def reactions(self):
        return copy.deepcopy(self._reactions)

    @property
    def diffusivity(self):
        return copy.deepcopy(self._diffusivity)

    def simulate(self, n_steps):
        log.info("Simulate for {} steps", n_steps)
        self._n_reactions = len(self._reactions)
        self._is_finalized = True
        for t in range(n_steps):
            log.debug("Step {}, system {}", t, self)
            possible_events = []
            cumulative = 0.
            # gather reaction events
            for i in range(self._n_boxes):
                for r in range(self._n_reactions):
                    propensity = self._reactions[r].propensity(self._state[i], i)
                    if propensity > 0.:
                        cumulative += propensity
                        delta = self._reactions[r].stoichiometric_delta
                        possible_events.append(ReactionEvent(i, delta, cumulative))
            # gather diffusion events
            for s in range(self._n_species):
                for i in range(self._n_boxes):
                    for j in range(self._n_boxes):
                        if i != j:
                            propensity = self._diffusivity[s][i, j] * self._state[i, s]
                            if propensity > 0.:
                                cumulative += propensity
                                delta = np.zeros(self._n_boxes, dtype=np.int)
                                delta[i] = -1
                                delta[j] = +1
                                possible_events.append(DiffusionEvent(s, delta, cumulative))

            if cumulative == 0.:
                log.info("No events possible / system is frustrated, at step {}", t)
                return

            # draw time and cumulative value
            event_time = (1. / cumulative) * np.log(1. / np.random.random())
            rnd = np.random.random() * cumulative

            # find event corresponding to rnd, that shall be performed
            cumulative_list = [x.cumulative_propensity for x in possible_events]
            event_idx = bisect.bisect_right(cumulative_list, rnd)
            event = possible_events[event_idx]
            log.debug("Cumulative list {}", cumulative_list)
            log.debug("Random number {}", rnd)
            log.debug("Event index for random number {}", event_idx)
            log.debug("Performing event {}", event)

            # save event and time to sequence
            self._event_list.append(event)
            self._time_list.append(self._time + event_time)

            # update system and save state
            self._time += event_time
            event.perform(self._state)
            self._state_list.append(np.copy(self._state))

    @property
    def sequence(self):
        return self._event_list, self._time_list, self._state_list

    # @todo move this into a result(event sequence)-object
    def convert_events_to_time_series(self, time_step=None, n_frames=None):
        """
        Convert to an equidistant time series. Therefore either a time_step or
        a number of frames must be give.

        For usage with readdy_learn (w/o diffusion as of 2017-06-29), do: counts = np.sum(result, axis=1)

        :param time_step: the temporal difference between each frame
        :param n_frames: the number of frames
        :return: a trajectory of shape (n_frames, n_boxes, n_species)
        """
        if not ((time_step is not None) ^ (n_frames is not None)):
            raise RuntimeError("Either time_step (x)or n_frames must be given")
        if len(self._time_list) < 2:
            raise RuntimeError("Sample some events first")
        if n_frames:
            time_step = (self._time_list[-1] - self._time_list[0]) / float(n_frames)
        else:
            n_frames = math.ceil((self._time_list[-1] - self._time_list[0]) / time_step)

        result = np.zeros((n_frames, self._n_boxes, self._n_species), dtype=np.int)
        times = np.linspace(self._time_list[0], self._time_list[-1], n_frames)
        current_t = self._time_list[0]
        last_passed_event_time_idx = 0
        result[0, :, :] = self._init_state
        for t in range(1, n_frames):
            # increment the time and find out how many events we have passed
            current_t += time_step
            n_passed_events = 0
            if current_t > self._time_list[last_passed_event_time_idx + 1]:
                n_passed_events = 1
                # if we have skipped one event we might have skipped another
                while current_t > self._time_list[last_passed_event_time_idx + n_passed_events + 1]:
                    n_passed_events += 1
            last_passed_event_time_idx += n_passed_events
            result[t] = self._state_list[last_passed_event_time_idx]
        return result, times
