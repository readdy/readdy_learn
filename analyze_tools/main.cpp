#include <pybind11/pybind11.h>
#include "src/cell_linked_list.h"
#include "src/logger.h"

namespace py = pybind11;

PYBIND11_PLUGIN(analyze_tools) {
    py::module module("analyze_tools", "analyze tools module");

    analyze_tools::cell_linked_list::export_to_python(module);
    analyze_tools::log::export_to_python(module);

    return module.ptr();
}