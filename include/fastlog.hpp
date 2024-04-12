//
// Created by Frank on 9/8/2022.
// Speeds up log calculations using a simple cache. Based on cache.hh and cache.cc in the graph-tool library
// (see https://git.skewed.de/count0/graph-tool/-/tree/master/src/graph/inference/support)
//

#ifndef DISTRIBUTEDSBP_FASTLOG_HPP
#define DISTRIBUTEDSBP_FASTLOG_HPP

#include <cmath>
#include <vector>

#include "args.hpp"

extern std::vector<double> fastlog_cache;

/// Initializes the cache with logs of small values.
void init_fastlog(size_t x);

inline double fastlog(size_t x) {
    if (x == 0) return 0.0;
    if (fastlog_cache.size() < args.cachesize) {
        init_fastlog(args.cachesize);
    }
    if (x >= fastlog_cache.size()) {
        return logf(double(x));
    }
    return fastlog_cache[x];
}

#endif //DISTRIBUTEDSBP_FASTLOG_HPP
