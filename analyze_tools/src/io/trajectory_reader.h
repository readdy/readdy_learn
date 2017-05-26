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
 * @file trajectory_reader.h
 * @brief << brief description >>
 * @author clonker
 * @date 26.05.17
 * @copyright GNU Lesser General Public License v3.0
 */

#pragma once

#include <readdy/io/io.h>
#include <readdy/model/observables/io/TrajectoryEntry.h>
#include <readdy/model/observables/io/Types.h>

namespace analyze_tools {
namespace io {

template<typename T>
inline void read(readdy::io::Group& group, const std::string& ds_name, std::vector<T> &array, readdy::io::DataSetType memoryType, readdy::io::DataSetType fileType) {
    auto name = readdy::log::console()->name();
    readdy::log::debug("name: {}", name);
    readdy::io::blosc_compression::initialize();

    const auto n_array_dims = 1 + readdy::io::util::n_dims<T>::value;
    auto hid = H5Dopen2(group.hid(), ds_name.data(), H5P_DEFAULT);

    readdy::io::DataSpace memorySpace (H5Dget_space(hid));

    const auto ndim = memorySpace.ndim();

    //if(ndim != n_array_dims) {
    //    log::error("wrong dimensionality: {} != {}", ndim, n_array_dims);
    //    throw std::invalid_argument("wrong dimensionality when attempting to read a data set");
    //}

    const auto dims = memorySpace.dims();
    std::size_t required_length = 1;
    for(const auto dim : dims) {
        readdy::log::error("dim len = {}", dim);
        required_length *= dim;
    }
    readdy::log::error("required length = {}", required_length);
    array.resize(required_length);

    auto result = H5Dread(hid, memoryType.hid(), H5S_ALL, H5S_ALL, H5P_DEFAULT, array.data());

    if(result < 0) {
        readdy::log::error("Failed reading result!");
        H5Eprint(H5Eget_current_stack(), stderr);
    }

    H5Dclose(hid);

    //for(std::size_t d = 0; d < ndim-1; ++d) {
    //    for(auto& sub_arr : array) {
    //        sub_arr.resize(dims[1]);
    //    }
    //}

    // todo reshape array to dims
}


}
}