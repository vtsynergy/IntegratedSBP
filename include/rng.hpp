//
// Created by Frank on 12/20/2022.
//

#ifndef SBP_RNG_HPP
#define SBP_RNG_HPP

#include "omp.h"
#include <random>
#include "pcg_random.hpp"

namespace rng {

typedef pcg32 Gen;
//typedef std::mt19937 Gen;

extern std::vector<Gen> generators;
extern std::vector<std::uniform_real_distribution<double>> distributions;

double generate();

Gen &generator();

void init_generators();

}

#endif //SBP_RNG_HPP
