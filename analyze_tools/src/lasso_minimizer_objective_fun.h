/********************************************************************
 * Copyright © 2016 Computational Molecular Biology Group,          * 
 *                  Freie Universität Berlin (GER)                  *
 *                                                                  *
 * This file is part of ReaDDy.                                     *
 *                                                                  *
 * ReaDDy is free software: you can redistribute it and/or modify   *
 * it under the terms of the GNU Lesser General Public License as   *
 * published by the Free Software Foundation, either version 3 of   *
 * the License, or (at your option) any later version.              *
 *                                                                  *
 * This program is distributed in the hope that it will be useful,  *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of   *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the    *
 * GNU Lesser General Public License for more details.              *
 *                                                                  *
 * You should have received a copy of the GNU Lesser General        *
 * Public License along with this program. If not, see              *
 * <http://www.gnu.org/licenses/>.                                  *
 ********************************************************************/


/**
 * << detailed description >>
 *
 * @file lasso_minimizer_objective_fun.h
 * @brief << brief description >>
 * @author clonker
 * @date 16.05.17
 * @copyright GNU Lesser General Public License v3.0
 */

#pragma once

#include <pybind11/numpy.h>

#include "logger.h"

namespace analyze_tools {

inline double theta_norm_squared(const pybind11::array_t<double, 0> &theta) {
    double result = 0;
    const auto n_timesteps = theta.shape()[0];
    const auto n_reactions = theta.shape()[1];
    const auto n_species = theta.shape()[2];

    for (std::size_t t = 0; t < n_timesteps; ++t) {
        for (std::size_t s = 0; s < n_species; ++s) {
            for (std::size_t r = 0; r < n_reactions; ++r) {
                const auto tval = theta.at(t,r,s);
                result += tval * tval;
            }
        }
    }
    return result;
}

inline double lasso_cost_fun(const pybind11::array_t<double, 0> &propensities, const double alpha,
                             const pybind11::array_t<double, 0> &theta, const pybind11::array_t<double, 0> &dX) {
    double result = 0;
    if(theta.ndim() != 3) {
        throw std::invalid_argument("invalid dims");
    }
    const auto n_timesteps = theta.shape()[0];
    const auto n_reactions = theta.shape()[1];
    const auto n_species = theta.shape()[2];
    for (std::size_t t = 0; t < n_timesteps; ++t) {
        for (std::size_t s = 0; s < n_species; ++s) {
            auto x = dX.at(t, s);
            for (std::size_t r = 0; r < n_reactions; ++r) {
                x -= propensities.at(r) * theta.at(t, r, s);
            }
            result += x * x;
        }
    }
    double regulator = 0;
    for (std::size_t r = 0; r < n_reactions; ++r) {
        regulator += std::abs(propensities.at(r));
    }
    regulator *= alpha;
    return result + regulator;
}
}
