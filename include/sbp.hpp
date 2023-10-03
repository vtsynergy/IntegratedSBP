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
/**
 * The stochastic block blockmodeling module.
 */
#ifndef SBP_SBP_HPP
#define SBP_SBP_HPP

#include <omp.h>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "blockmodel/blockmodel_triplet.hpp"
#include "graph.hpp"

namespace sbp {

// The time taken to finetune the partition.
extern double finetune_time;
/// The total amount of time spent community detection, to be dynamically updated during execution.
extern double total_time;
/// The total number of island vertices (across all MPI ranks, if applicable)
extern long total_num_islands;

/// Stores intermediate information for later printing.
struct intermediate {
    double iteration;
    double mdl;
    double normalized_mdl_v1;
    double modularity;
    long mcmc_iterations;
    double mcmc_time;
    double mcmc_sequential_time;
    double mcmc_parallel_time;
    double mcmc_vertex_move_time;
    ulong mcmc_moves;
    double block_merge_time;
    double block_merge_loop_time;
    double blockmodel_build_time;
    double finetune_time;
    double load_balancing_time = 0.0;
    double sort_time;
    double access_time;
    double update_assignment;
    double total_time;
};

/// Adds intermediate results to be later saved in a CSV file.
void add_intermediate(double iteration, const Graph &graph, double modularity, double mdl);

std::vector<intermediate> get_intermediates();

/// Performs community detection on the provided graph, using the stochastic block partitioning algorithm
Blockmodel stochastic_block_partition(Graph &graph, Args &args, bool divide_and_conquer = false);

/// Returns true if the exit condition is reached based on the provided blockmodels
bool done_blockmodeling(Blockmodel &blockmodel, BlockmodelTriplet &blockmodel_triplet, long min_num_blocks = 0);

} // namespace sbp

#endif // SBP_SBP_HPP
