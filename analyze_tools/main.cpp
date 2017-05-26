#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "src/cell_linked_list.h"
#include "src/logger.h"
#include "src/lasso_minimizer_objective_fun.h"

namespace py = pybind11;

namespace analyze_tools {
namespace log {
inline static void export_to_python(pybind11::module &m) {
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
    m_log.def("trace", [](const std::string &message) { trace(message); });
    m_log.def("debug", [](const std::string &message) { debug(message); });
    m_log.def("info", [](const std::string &message) { info(message); });
    m_log.def("warn", [](const std::string &message) { warn(message); });
    m_log.def("error", [](const std::string &message) { error(message); });
    m_log.def("critical", [](const std::string &message) { critical(message); });
}

}

namespace opt {
inline static void export_to_python(py::module &m) {
    using namespace py::literals;
    auto module = m.def_submodule("opt");
    module.def("lasso_minimizer_objective_fun", &lasso_cost_fun, "propensities"_a, "alpha"_a, "theta"_a, "dX"_a,
               "prefactor"_a = -1.);
    module.def("elastic_net_objective_fun", &elastic_net_objective_function, "propensities"_a, "alpha"_a, "l1_ratio"_a,
               "theta"_a, "dX"_a, "prefactor"_a = -1.);
    module.def("theta_norm_squared", &theta_norm_squared);
    module.def("score", &score);
}

}
}

PYBIND11_PLUGIN(analyze_tools) {
    py::module module("analyze_tools", "analyze tools module");

    pybind11::class_<analyze_tools::cell_linked_list>(module, "CellLinkedList").def(pybind11::init<std::string>());
    analyze_tools::log::export_to_python(module);
    analyze_tools::opt::export_to_python(module);
    return module.ptr();
}
