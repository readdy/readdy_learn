#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "src/cell_linked_list.h"
#include "src/lasso_minimizer_objective_fun.h"

namespace py = pybind11;

namespace analyze_tools {

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
    analyze_tools::opt::export_to_python(module);
    return module.ptr();
}
