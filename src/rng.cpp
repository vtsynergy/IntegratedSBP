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
