#pragma once
#include "generator.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Samples a generator function at equidistant uniform time intervals.
 * 
 */
struct UniformSampler
{
    const double frequency;

    /**
     * @brief Create a new uniform sampler with a specific sample rate.
     * 
     * @param frequency The sample frequency.
     */
    explicit UniformSampler(const double frequency);

    /**
     * @brief Sample a generator over a duration.
     * 
     * @param generator The generator function to sample. 
     * @param duration The duration over which to sample the function.
     * @return A python list containing the sample points.
     */
    py::list sample(const Generator& generator, const double duration = 1) const;
};
