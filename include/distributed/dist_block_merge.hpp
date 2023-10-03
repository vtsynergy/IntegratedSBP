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