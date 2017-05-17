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
Created on 17.05.17

@author: clonker
"""

import numpy as np


def preprocess_data(X, y, normalize):
    """
    X = (X - X_offset) / X_scale
    """

    X_offset = np.average(X, axis=0)
    X -= X_offset
    if normalize:
        from sklearn.preprocessing.data import normalize
        X, X_scale = normalize(X, axis=0, copy=False, return_norm=True)
    else:
        X_scale = np.ones(X.shape[1])
    y_offset = np.average(y, axis=0)
    y = y - y_offset

    return X, y, X_offset, y_offset, X_scale
