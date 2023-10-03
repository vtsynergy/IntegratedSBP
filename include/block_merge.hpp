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
 * The block merge phase of the stochastic block blockmodeling module.
 */
#ifndef CPPSBP_BLOCK_MERGE_HPP
#define CPPSBP_BLOCK_MERGE_HPP

#include <limits>
#include <numeric>
#include <random>

// #include <omp.h>
#include "common.hpp"
#include "blockmodel/blockmodel.hpp"
// #include "blockmodel/sparse/boost_mapped_matrix.hpp"
#include "blockmodel/sparse/csparse_matrix.hpp"
#include "blockmodel/sparse/typedefs.hpp"
#include "graph.hpp"
#include "utils.hpp"
#include "typedefs.hpp"

namespace block_merge {

/// The total amount of time spent performing block merges, to be dynamically updated during execution.
extern double BlockMerge_time;

/// The total amount of time spent in the main parallelizable loop of the block merge iterations, to by dynamically
/// updated during execution.
extern double BlockMerge_loop_time;

/// The time taken to sort potential block merges.
extern double BlockMerge_sort_time;

static const long NUM_AGG_PROPOSALS_PER_BLOCK = 10;  // Proposals per block

typedef struct proposal_evaluation_t {
    long proposed_block;
    double delta_entropy;
} ProposalEvaluation;

/// Performs the block merges with the highest change in entropy/MDL, recalculating change in entropy before each
/// merge to account for dependencies between merges. This function modified the blockmodel.
void carry_out_best_merges_advanced(Blockmodel &blockmodel, const std::vector<double> &delta_entropy_for_each_block,
                                    const std::vector<long> &best_merge_for_each_block, long num_edges);

/// Returns the potential changes to the blockmodel if `current_block` was merged into `proposed_block`.
Delta blockmodel_delta(long current_block, long proposed_block, const Blockmodel &blockmodel);

/// Computes the new edge counts for the affected blocks (communities) under a proposed block merge.
EdgeCountUpdates edge_count_updates(std::shared_ptr<ISparseMatrix> blockmodel, long current_block, long proposed_block,
                                    EdgeWeights &out_blocks, EdgeWeights &in_blocks);

/// Fills the new edge counts for the affected blocks (communities) under a proposed block merge.
/// Results are stored as sparse vectors (unordered_maps)
void edge_count_updates_sparse(ISparseMatrix *blockmodel, long current_block, long proposed_block,
                               EdgeWeights &out_blocks, EdgeWeights &in_blocks, SparseEdgeCountUpdates &updates);

/// Merges entire blocks (communities) in blockmodel together
Blockmodel &merge_blocks(Blockmodel &blockmodel, const Graph &graph, long num_edges);

/// Proposes a merge for current_block based on the current blockmodel state
ProposalEvaluation propose_merge(long current_block, long num_edges, Blockmodel &blockmodel,
                                 std::vector<long> &block_assignment);

/// Proposes a merge for current_block based on the current blockmodel state, using sparse intermediate structures
ProposalEvaluation propose_merge_sparse(long current_block, long num_edges, const Blockmodel &blockmodel,
                                        std::unordered_map<long, bool> &past_proposals);

} // namespace block_merge

#endif // CPPSBP_BLOCK_MERGE_HPP
