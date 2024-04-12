//
// Created by Frank on 12/29/2023.
//

#include "fastlgamma.hpp"

#include <omp.h>

std::vector<double> fastlgamma_cache;

void init_fastlgamma(size_t x) {
    #pragma omp critical (fastlgamma)
    {
        if (x >= fastlgamma_cache.size()) {
            fastlgamma_cache.resize(x + 1);
            fastlgamma_cache[0] = std::numeric_limits<double>::infinity();
            for (size_t i = 1; i < fastlgamma_cache.size(); ++i) {
                fastlgamma_cache[i] = lgamma(double(i));
            }
        }
//        size_t old_size = __lgamma_cache.size();
//        if (x >= old_size)
//        {
//            __lgamma_cache.resize(x + 1);
//            if (old_size == 0)
//                __lgamma_cache[0] = numeric_limits<double>::infinity();
//            for (size_t i = std::max(old_size, size_t(1));
//                 i < __lgamma_cache.size(); ++i)
//                __lgamma_cache[i] = lgamma(i);
//        }
    }
}
