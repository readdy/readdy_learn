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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace analyze_tools {
namespace opt {
namespace py = pybind11;
using input_array = py::array_t<double, py::array::c_style>;

double theta_norm_squared(const input_array &theta);

double score(const input_array &propensities, const input_array &theta, const input_array &dX);

void least_squares_function(input_array &result, const input_array &propensities, const input_array &theta,
                            const input_array &dX);

input_array elastic_net_objective_function_jac(const input_array &propensities,
                                               const double alpha, const double l1_ratio, const input_array &theta,
                                               const input_array &dX);

double elastic_net_objective_function(const input_array &propensities, const double alpha, const double l1_ratio,
                                      const input_array &theta, const input_array &dX);

double lasso_cost_fun(const input_array &propensities, const double alpha,
                      const input_array &theta, const input_array &dX);


}
}
