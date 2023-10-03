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
 * Functions relating to calculating blockmodel entropy.
 */

#include "blockmodel/blockmodel.hpp"
#include "distributed/two_hop_blockmodel.hpp"
#include "common.hpp"
#include "delta.hpp"
#include "utils.hpp"

namespace entropy {

/// Computes the change in entropy under a proposed block merge.
double block_merge_delta_mdl(long current_block, long proposal, long num_edges, const Blockmodel &blockmodel,
                             EdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees);

/// Computes the change in entropy under a proposed block merge using sparse intermediate structures.
double block_merge_delta_mdl(long current_block, long proposal, long num_edges, const Blockmodel &blockmodel,
                             SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees);

/// Computes the change in entropy under a proposed block merge using changes to the blockmodel.
double block_merge_delta_mdl(long current_block, const Blockmodel &blockmodel, const Delta &delta,
                             common::NewBlockDegrees &block_degrees);

/// Computes the change in entropy under a proposed block merge using changes to the blockmodel. This method should
/// be preferred in almost all cases.
double block_merge_delta_mdl(long current_block, utils::ProposalAndEdgeCounts proposal, const Blockmodel &blockmodel,
                             const Delta &delta);

/// Computes the change in blockmodel minimum description length when a vertex moves from `current_block` to `proposal`.
/// Uses a dense version of `updates` to the blockmodel, and requires pre-calculated updated `block_degrees`.
double delta_mdl(long current_block, long proposal, const Blockmodel &blockmodel, long num_edges,
                 EdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees);

/// Computes the change in blockmodel minimum description length when a vertex moves from `current_block` to `proposal`.
/// Uses a sparse version of `updates` to the blockmodel, and requires pre-calculated updated `block_degrees`.
double delta_mdl(long current_block, long proposal, const Blockmodel &blockmodel, long num_edges,
                 SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees);

/// Computes the change in blockmodel minimum description length when a vertex moves from one block to another. Uses
/// changes to the blockmodel, stored in `delta`, to perform the computation, and does not require pre-calculated
/// updated block_degrees. This method should be preferred in almost all cases.
double delta_mdl(const Blockmodel &blockmodel, const Delta &delta, const utils::ProposalAndEdgeCounts &proposal);

/// Computes the Hastings correction using dense vectors.
double hastings_correction(const Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           utils::ProposalAndEdgeCounts &proposal, EdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees);

/// Computes the Hastings correction using sparse vectors.
double hastings_correction(const Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           utils::ProposalAndEdgeCounts &proposal, SparseEdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees);

/// Computes the hastings correction using the blockmodel deltas under the proposed vertex move.
double hastings_correction(long vertex, const Graph &graph, const Blockmodel &blockmodel, const Delta &delta,
                           long current_block, const utils::ProposalAndEdgeCounts &proposal);

/// Calculates the minimum description length of `blockmodel` for a directed graph with `num_vertices` vertices and
/// `num_edges` edges.
double mdl(const Blockmodel &blockmodel, long num_vertices, long num_edges);

/// Computes the normalized minimum description length using `null_mdl_v1`.
double normalize_mdl_v1(double mdl, long num_edges);

/// Computes the normalized minimum description length using `null_mdl_v2`.
double normalize_mdl_v2(double mdl, long num_vertices, long num_edges);

/// Computes the minimum description length of the null model with only one block.
double null_mdl_v1(long num_edges);

/// Computes the minimum description length of the null model with as many blocks as there are vertices.
double null_mdl_v2(long num_vertices, long num_edges);

// TODO: add an undirected mdl
// TODO: add undirected delta_mdl functions

namespace dist {

/// Computes the overall entropy of the given blockmodel for a directed graph.
double mdl(const TwoHopBlockmodel &blockmodel, long num_vertices, long num_edges);

// TODO: add an undirected distributed mdl

// TODO: handle case when number of edges/vertices >= 2.15 billion (long --> long)

}

}