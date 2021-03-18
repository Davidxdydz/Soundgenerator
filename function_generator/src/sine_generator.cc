#include "sine_generator.hpp"
#define _USE_MATH_DEFINES
#include <cmath>

SineGenerator::SineGenerator(const double frequency) : frequency(frequency) {}

double SineGenerator::operator()(const double x) const
{
    return sin(x * 2 * M_PI * frequency);
}