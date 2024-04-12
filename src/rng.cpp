//
// Created by Frank on 12/20/2022.
//

#include "rng.hpp"

#include "args.hpp"

#include <iostream>

namespace rng {

std::vector<Gen> generators;
std::vector<std::uniform_real_distribution<double>> distributions;

double generate() {
//    if (generators.size() < omp_get_max_threads()) {
//        init_generators();
//    }
    long thread_id = omp_get_thread_num();
    ulong generated_long = generators[thread_id]();
    return double(generated_long) / double(generators[thread_id].max());
}

Gen &generator() {
//    if (generators.size() < omp_get_max_threads()) {
//        init_generators();
//    }
    Gen &generator = generators[omp_get_thread_num()];
    return generator;
}

void init_generators() {
    pcg_extras::seed_seq_from<std::random_device> seed_source;
//    std::random_device seeder;
    long num_threads = args.threads;
    for (long i = 0; i < num_threads; ++i) {
//        Gen generator(seeder());
        Gen generator(seed_source);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        generators.push_back(generator);
        distributions.push_back(distribution);
    }
}

} // namespace rng
