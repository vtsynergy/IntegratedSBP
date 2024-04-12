//
// Created by Frank on 9/8/2022.
// Speeds up log calculations using a simple cache. Based on cache.hh and cache.cc in the graph-tool library
// (see https://git.skewed.de/count0/graph-tool/-/tree/master/src/graph/inference/support)
//

#include "fastlog.hpp"

#include <omp.h>

std::vector<double> fastlog_cache;

void init_fastlog(size_t x) {
    #pragma omp critical (fastlog)
    {
        if (x >= fastlog_cache.size()) {
            fastlog_cache.resize(x + 1);
            for (size_t i = 0; i < fastlog_cache.size(); ++i) {
                fastlog_cache[i] = logf(double(i));
            }
        }
    }
}
