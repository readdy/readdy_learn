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
 * @file logger.h
 * @brief << brief description >>
 * @author clonker
 * @date 10.05.17
 * @copyright GNU Lesser General Public License v3.0
 */

#pragma once

#include <utility>
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>

namespace analyze_tools {
namespace log {
std::shared_ptr<spdlog::logger> console();

template<typename... Args>
void trace(Args &&... args) {
    console()->trace(std::forward<Args>(args)...);
}

template<typename... Args>
void debug(Args &&... args) {
    console()->debug(std::forward<Args>(args)...);
}

template<typename... Args>
void critical(Args &&... args) {
    console()->critical(std::forward<Args>(args)...);
}

template<typename... Args>
void warn(Args &&... args) {
    console()->warn(std::forward<Args>(args)...);
}

template<typename... Args>
void error(Args &&... args) {
    console()->error(std::forward<Args>(args)...);
}

template<typename... Args>
void info(Args &&... args) {
    console()->info(std::forward<Args>(args)...);
}

inline static void export_to_python(pybind11::module& m) {
    auto m_log = m.def_submodule("log", "logging submodule");
    m_log.def("set_level", [](const std::string &level) -> void {
        console()->set_level([&level] {
            if (level == "trace") {
                return spdlog::level::trace;
            } else if (level == "debug") {
                return spdlog::level::debug;
            } else if (level == "info") {
                return spdlog::level::info;
            } else if (level == "warn") {
                return spdlog::level::warn;
            } else if (level == "err" || level == "error") {
                return spdlog::level::err;
            } else if (level == "critical") {
                return spdlog::level::critical;
            } else if (level == "off") {
                return spdlog::level::off;
            }
            warn("Did not select a valid logging level, setting to debug!");
            return spdlog::level::debug;
        }());
    }, "Function that sets the logging level. Possible arguments: \"trace\", \"debug\", \"info\", \"warn\", "
                       "\"err\", \"error\", \"critical\", \"off\".");
    m_log.def("trace", [](const std::string& message) { trace(message); });
    m_log.def("debug", [](const std::string& message) { debug(message); });
    m_log.def("info", [](const std::string& message) { info(message); });
    m_log.def("warn", [](const std::string& message) { warn(message); });
    m_log.def("error", [](const std::string& message) { error(message); });
    m_log.def("critical", [](const std::string& message) { critical(message); });
}
}
}
