/// ====================================================================================================================
/// Part of the accelerated Stochastic Block Partitioning (SBP) project.
/// Copyright (C) Virginia Polytechnic Institute and State University, 2023. All Rights Reserved.
///
/// This software is provided as-is. Neither the authors, Virginia Tech nor Virginia Tech Intellectual Properties, Inc.
/// assert, warrant, or guarantee that the software is fit for any purpose whatsoever, nor do they collectively or
/// individually accept any responsibility or liability for any action or activity that results from the use of this
/// software.  The entire risk as to the quality and performance of the software rests with the user, and no remedies
/// shall be provided by the authors, Virginia Tech or Virginia Tech Intellectual Properties, Inc.
/// This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
/// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
/// details.
/// You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to
/// the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.
///
/// Author: Frank Wanye
/// ====================================================================================================================
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
    if (fastlog_cache.size() < args.cachesize) {
        init_fastlog(args.cachesize);
    }
    if (x >= fastlog_cache.size()) {
        return logf(double(x));
    }
    return fastlog_cache[x];
}

#endif //DISTRIBUTEDSBP_FASTLOG_HPP
