#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_PLUGIN(analyze_tools) {
    py::module module("analyze_tools");

    return module.ptr();
}