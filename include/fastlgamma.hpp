//
// Created by Frank on 12/29/2023.
//

#ifndef DISTRIBUTEDSBP_FASTLGAMMA_HPP
#define DISTRIBUTEDSBP_FASTLGAMMA_HPP

#include <cmath>
#include <vector>

#include "args.hpp"

extern std::vector<double> fastlgamma_cache;

/// Initializes the cache with gammas of small values.
void init_fastlgamma(size_t x);

inline double fastlgamma(size_t x) {
    if (fastlgamma_cache.size() < args.cachesize) {
        init_fastlgamma(args.cachesize);
    }
    if (x >= fastlgamma_cache.size()) {
        return lgamma(double(x));
    }
    return fastlgamma_cache[x];
}

#endif //DISTRIBUTEDSBP_FASTLGAMMA_HPP
