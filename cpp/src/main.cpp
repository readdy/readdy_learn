#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "lasso_minimizer_objective_fun.h"

namespace py = pybind11;

namespace analyze_tools {

namespace opt {
inline static void export_to_python(py::module &m) {
    using namespace py::literals;
    auto module = m.def_submodule("opt");
    module.def("lasso_minimizer_objective_fun", &lasso_cost_fun, "propensities"_a, "alpha"_a, "theta"_a, "dX"_a);
    module.def("elastic_net_objective_fun", &elastic_net_objective_function, "propensities"_a, "alpha"_a, "l1_ratio"_a,
               "theta"_a, "dX"_a);
    module.def("elastic_net_objective_fun_jac", &elastic_net_objective_function_jac, "propensities"_a, "alpha"_a,
               "l1_ratio"_a, "theta"_a, "dX"_a);
    module.def("theta_norm_squared", &theta_norm_squared);
    module.def("score", &score);
    module.def("least_squares_function", &least_squares_function);
}

}
}

using kmc_result_array = py::array_t<std::uint32_t, py::array::c_style>;
using kmc_state_array = py::array_t<std::uint32_t, py::array::c_style>;
using kmc_times_array = py::array_t<double, py::array::c_style>;

static void convert_kmc(kmc_result_array &result, const kmc_times_array &times, const kmc_times_array &times_list, const kmc_state_array &state_list) {
    std::size_t state = 0;

    auto nstates = (std::size_t) state_list.shape()[0];
    auto nframes = (std::size_t) result.shape()[0];
    auto nboxes = (std::size_t) result.shape()[1];
    auto nspecies = (std::size_t) result.shape()[2];

    for(std::size_t ix = 0; ix < (std::size_t) times.shape()[0]; ++ix) {
        auto t = times.at(ix);
        if(t <= times_list.at(state)) {
            for (std::size_t s = 0; s < nspecies; ++s) {
                result.mutable_at(state, 0, s) = state_list.at(state, 0, s);
            }
        } else {
            while(state < nstates && t > times_list.at(state)) {
                ++state;
            }
            for (std::size_t s = 0; s < nspecies; ++s) {
                result.mutable_at(state, 0, s) = state_list.at(state, 0, s);
            }
        }
    }
}

PYBIND11_MODULE(analyze_tools, m) {
    analyze_tools::opt::export_to_python(m);
    auto tools = m.def_submodule("tools");
    tools.def("convert_kmc", &convert_kmc);
}
