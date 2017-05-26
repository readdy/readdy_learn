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
Created on 23.05.17

@author: clonker
"""

import unittest
import h5py
from analyze_tools.py_analyze_tools.tools import TrajectoryConfig

import analyze_tools.analyze_tools as at

class Tests(unittest.TestCase):

    def test_load_trajectory_config(self):
        with h5py.File("generate/simple_trajectory_4.h5") as f:
            cfg = TrajectoryConfig(f)

    def test_read_trajectory(self):
        foo = at.CellLinkedList("foo")

if __name__ == '__main__':
    unittest.main()