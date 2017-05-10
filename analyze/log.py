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

import analyze_tools.analyze_tools.log as log


def set_level(level):
    log.set_level(level)


def trace(msg):
    log.trace(msg)


def debug(msg):
    log.debug(msg)


def info(msg):
    log.info(msg)


def warn(msg):
    log.warn(msg)


def error(msg):
    log.error(msg)


def critical(msg):
    log.critical(msg)
