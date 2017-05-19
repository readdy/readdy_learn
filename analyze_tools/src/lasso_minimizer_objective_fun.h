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
namespace opt {
namespace py = pybind11;
using input_array = py::array_t<double, 0>;

inline double theta_norm_squared(const input_array &theta) {
    double result = 0;
    const auto n_timesteps = theta.shape()[0];
    const auto n_reactions = theta.shape()[1];
    const auto n_species = theta.shape()[2];

    for (std::size_t t = 0; t < n_timesteps; ++t) {
        for (std::size_t s = 0; s < n_species; ++s) {
            for (std::size_t r = 0; r < n_reactions; ++r) {
                const auto tval = theta.at(t, r, s);
                result += tval * tval;
            }
        }
    }
    return result;
}

inline double score(const input_array &propensities, const input_array &theta, const input_array &dX) {
    double result = 0;
    if (theta.ndim() != 3) {
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
    return std::sqrt(result);
}

inline double elastic_net_objective_function(const input_array &propensities, const double alpha, const double l1_ratio,
                                             const input_array &theta, const input_array &dX,
                                             const double prefactor = -1) {
    double result = 0;
    if (theta.ndim() != 3) {
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
    if (prefactor >= 0) {
        result *= prefactor;
    } else {
        result *= 1. / (2. * n_timesteps * n_species);
    }
    double regulator = 0;
    {
        double regulator_l1 = 0;
        for (std::size_t r = 0; r < n_reactions; ++r) {
            regulator_l1 += std::abs(propensities.at(r));
        }
        regulator_l1 *= alpha * l1_ratio;
        regulator += regulator_l1;
    }
    if (l1_ratio < 1.0) {
        double l2_norm_squared = 0;
        for (std::size_t r = 0; r < n_reactions; ++r) {
            l2_norm_squared += propensities.at(r) * propensities.at(r);
        }
        l2_norm_squared *= 0.5 * alpha * (1 - l1_ratio);
        regulator += l2_norm_squared;
    }
    return result + regulator;
}

inline double lasso_cost_fun(const input_array &propensities, const double alpha,
                             const input_array &theta, const input_array &dX, const double prefactor = -1) {
    return elastic_net_objective_function(propensities, alpha, 1.0, theta, dX, prefactor);
}

inline static void export_to_python(py::module &m) {
    using namespace py::literals;
    auto module = m.def_submodule("opt");
    module.def("lasso_minimizer_objective_fun", &lasso_cost_fun, "propensities"_a, "alpha"_a, "theta"_a, "dX"_a,
               "prefactor"_a = -1.);
    module.def("elastic_net_objective_fun", &elastic_net_objective_function, "propensities"_a, "alpha"_a, "l1_ratio"_a,
               "theta"_a, "dX"_a,
               "prefactor"_a = -1.);
    module.def("theta_norm_squared", &theta_norm_squared);
    module.def("score", &score);
}

}
}
