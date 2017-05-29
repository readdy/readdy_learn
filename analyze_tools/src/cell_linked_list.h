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
 * @file cell_linked_list.h
 * @brief << brief description >>
 * @author clonker
 * @date 10.05.17
 * @copyright GNU Lesser General Public License v3.0
 */

#pragma once

#include <readdy/io/io.h>
#include <readdy/model/observables/io/TrajectoryEntry.h>
#include <readdy/model/observables/io/Types.h>
#include <iostream>
#include "io/trajectory_reader.h"


namespace analyze_tools {
class cell_linked_list {
public:

    cell_linked_list(const std::string& fname) {
        std::cout << "fname=" << fname << std::endl;
        readdy::io::File f("/home/mho/Development/readdy/readdy/readdy/test/simple_trajectory.h5", readdy::io::File::Action::OPEN, readdy::io::File::Flag::READ_ONLY);
        auto& rootGroup = f.getRootGroup();
        auto traj = rootGroup.subgroup("readdy/trajectory");
        for(const auto ds : traj.contained_data_sets()) {
            readdy::log::error("ds: {}", ds);
        }
        // limits
        std::vector<std::size_t> limits;
        io::read(traj,"limits", limits, readdy::io::STDDataSetType<std::size_t>(), readdy::io::NativeDataSetType<std::size_t>());
        // records
        std::vector<readdy::model::observables::TrajectoryEntry> entries;
        readdy::model::observables::util::TrajectoryEntryMemoryType memoryType;
        readdy::model::observables::util::TrajectoryEntryFileType fileType;
        traj.read("records", entries, memoryType, fileType);

        for(std::size_t i = 0; i < limits.size(); i += 2) {
            auto begin = limits[i];
            auto end = limits[i+1];
            readdy::log::error("got n frames: {} = {} - {}", end - begin, end, begin);
            readdy::log::error("last entry: {}", entries[end-1]);
        }

        std::vector<readdy::time_step_type> timesteps;
        traj.read("time", timesteps);
        for(auto t : timesteps) {
            readdy::log::error("t = {}", t);
        }
    }


};
}
