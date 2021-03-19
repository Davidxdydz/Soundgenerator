#include <pybind11/pybind11.h>
#include "generator.hpp"
#include "sine_generator.hpp"
#include "saw_generator.hpp"
#include "uniform_sampler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(function_generator, m)
{
    m.doc() = "A simple function generator and sampling module";

    // Expose virtual generator base.
    // This has to be done to allow:
    //  1.) Python classes to inherit from this.
    //  2.) Enable automatic downcasting for c++ generators during runtime.
    py::class_<Generator>(m, "Generator");
    
    // Expose sinus generator.
    py::class_<SineGenerator, Generator>(m, "SineGenerator")
        .def(py::init<const double>(), "Construct a sine generator with specific frequency.")
        .def("__call__", &SineGenerator::operator(), "Sample sinus function for duration.", py::arg("t"))
        .def_readonly("frequency", &SineGenerator::frequency, "Frequency of sinus generator.");

    // Expose saw wave generator.
    py::class_<SawGenerator, Generator>(m, "SawGenerator")
        .def(py::init<const double>(), "Construct a saw generator with specific frequency.")
        .def("__call__", &SawGenerator::operator(), "Sample saw wave function for duration.", py::arg("t"))
        .def_readonly("frequency", &SawGenerator::frequency, "Frequency of saw wave generator.");

    // Expose uniform sampler.
    py::class_<UniformSampler>(m, "UniformSampler")
        .def(py::init<const double>(), "Construct a uniform sampler with a specific sample frequency.")
        .def("sample", &UniformSampler::sample, "Sample generator for given time.", py::arg("generator"), py::arg("duration") = 1)
        .def_readonly("frequency", &UniformSampler::frequency, "Frequency of sampler.");

}
