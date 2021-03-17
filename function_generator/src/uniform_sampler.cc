#include "uniform_sampler.hpp"

UniformSampler::UniformSampler(const double frequency) : frequency(frequency) {}

py::list UniformSampler::sample(const Generator &generator, const double duration) const
{
    // Make sure duration is positive
    if (duration < 0)
        throw std::invalid_argument("Duration must be positive!");

    // Make result list.
    auto result = py::list();

    // TODO: Run this with omp...?
    for (float x = 0; x < duration; x += 1 / frequency)
        result.append(generator(x));

    return result;
}