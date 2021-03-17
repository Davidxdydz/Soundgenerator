#pragma once
#include "generator.hpp"

/**
 * @brief Generates a sine wave.
 * 
 */
struct SineGenerator : Generator
{
    const double frequency;

    /**
     * @brief Construct a new sinus generator. 
     * 
     * @param frequency Guess.
     */
    explicit SineGenerator(const double frequency);

    /**
     * @brief Evaluate the function generator at a certain point.
     * 
     * @param x Point at which to evalute the generator.
     * @return sin(2 * x * pi * frequency) 
     */
    double operator()(const double x) const;
};