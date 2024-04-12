/**
 * The distributed block merge phase of the stochastic block blockmodeling module.
 */
#ifndef CPPSBP_DIST_BLOCK_MERGE_HPP
#define CPPSBP_DIST_BLOCK_MERGE_HPP

#include "block_merge.hpp"
#include "distributed/two_hop_blockmodel.hpp"

namespace block_merge::dist {

/// Merges entire blocks (communities) in blockmodel together in a distributed fashion.
TwoHopBlockmodel &merge_blocks(TwoHopBlockmodel &blockmodel, const Graph &graph);

}  // namespace block_merge::dist

#endif // CPPSBP_DIST_BLOCK_MERGE_HPP