#include <pybind11/embed.h>

int preparePybind11() {
    py::scoped_interpreter python{}; // uniform initialization intoduced in c++11
    py::module_::import("numpy");
}


