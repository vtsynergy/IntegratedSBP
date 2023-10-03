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
