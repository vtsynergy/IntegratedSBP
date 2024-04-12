/**
 * Structs and functions common to both the distributed block merge and finetune phases.
 */
#ifndef SBP_DIST_COMMON_HPP
#define SBP_DIST_COMMON_HPP

#include <vector>

#include "distributed/two_hop_blockmodel.hpp"
#include "typedefs.hpp"
#include "utils.hpp"

namespace common::dist {

// TODO: get rid of block_assignment, just use blockmodel?
utils::ProposalAndEdgeCounts propose_new_block(long current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                               const std::vector<long> &block_assignment,
                                               const TwoHopBlockmodel &blockmodel, bool block_merge);

} // namespace common::dist

#endif  // SBP_DIST_COMMON_HPP
