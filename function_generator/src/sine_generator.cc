#include "sine_generator.hpp"
#include <cmath>

SineGenerator::SineGenerator(const double frequency) : frequency(frequency) {}

double SineGenerator::operator()(const double t) const
{
    return sin(t * 2 * M_PI * frequency);
}