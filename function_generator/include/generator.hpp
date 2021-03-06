#pragma once

/**
 * @brief Virtual base for all function generators.
 * 
 */
struct Generator
{
    /**
     * @brief Evaluate the generator on a certain point. This call may not change
     *        the internal state of the generator.
     * 
     * @param t The time at which to perform evaluation.
     * @return The function at the current point.
     */
    virtual double operator()(const double t) const = 0;

    /**
     * @brief Destroy the Generator object
     * 
     */
    virtual ~Generator() = default;
};
