#include "sine_generator.hpp"
#include <cmath>

SineGenerator::SineGenerator(const double frequency) : frequency(frequency) {}

double SineGenerator::operator()(const double x) const
{
    return sin(x * 2 * M_PI * frequency);
}