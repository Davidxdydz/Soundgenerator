#include "saw_generator.hpp"
#define _USE_MATH_DEFINES
#include <cmath>

SawGenerator::SawGenerator(const double frequency) : frequency(frequency) {}

double SawGenerator::operator()(const double t) const
{
    return fmod(t * frequency, 1.0) * 2.0 - 1.0;
}