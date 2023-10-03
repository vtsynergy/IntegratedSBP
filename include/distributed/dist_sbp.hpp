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
 * The distributed stochastic block blockmodeling module.
 */
#ifndef SBP_DIST_SBP_HPP
#define SBP_DIST_SBP_HPP

#include "args.hpp"
#include "blockmodel.hpp"
#include "distributed/dist_blockmodel_triplet.hpp"
#include "distributed/two_hop_blockmodel.hpp"
#include "graph.hpp"
#include "sbp.hpp"

namespace sbp::dist {

// The amount of time taken to finetune the partition.
extern double finetune_time;

/// Adds intermediate results to be later saved in a CSV file.
void add_intermediate(double iteration, const Graph &graph, double modularity, double mdl);

std::vector<intermediate> get_intermediates();

/// Performs community detection on the provided graph using MPI, using the stochastic block partitioning algorithm
Blockmodel stochastic_block_partition(Graph &graph, Args &args, bool divide_and_conquer = false);

/// Returns true if the exit condition is reached based on the provided distributed blockmodels
bool done_blockmodeling(TwoHopBlockmodel &blockmodel, DistBlockmodelTriplet &blockmodel_triplet,
                        long min_num_blocks = 0);

} // namespace sbp::dist

#endif  // SBP_DIST_SBP_HPP