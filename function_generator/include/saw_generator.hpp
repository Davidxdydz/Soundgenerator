#pragma once
#include "generator.hpp"

/**
 * @brief Generates a saw wave.
 * 
 */
struct SawGenerator : Generator
{
    const double frequency;

    /**
     * @brief Construct a new saw wave generator. 
     * 
     * @param frequency Guess.
     */
    explicit SawGenerator(const double frequency);

    /**
     * @brief Evaluate the function generator at a certain point.
     * 
     * @param t Point at which to evalute the generator.
     * @return sin(2 * x * pi * frequency) 
     */
    double operator()(const double t) const;
};