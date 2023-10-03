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
 * Stores the distributed triplet of blockmodels needed for the fibonacci search.
 */
#ifndef SBP_DIST_BLOCKMODEL_TRIPLET_HPP
#define SBP_DIST_BLOCKMODEL_TRIPLET_HPP

#include "distributed/two_hop_blockmodel.hpp"

class DistBlockmodelTriplet {

public:
    DistBlockmodelTriplet() : optimal_num_blocks_found(false) {}
    /// TODO
    bool optimal_num_blocks_found;
    /// TODO
    TwoHopBlockmodel &get(long i) { return this->blockmodels[i]; }
    /// TODO
    TwoHopBlockmodel get_next_blockmodel(TwoHopBlockmodel &old_blockmodel);
    /// TODO
    bool golden_ratio_not_reached();
    /// TODO
    bool is_done();
    /// TODO
    void update(TwoHopBlockmodel &blockmodel);
    /// TODO
    void status();

private:
    /// Blockmodels arranged in order of decreasing number of blocks.
    /// If the first blockmodel is empty, then the golden ratio bracket has not yet been established.
    /// TODO
    TwoHopBlockmodel blockmodels[3];
    /// TODO
    long lower_difference();
    /// TODO
    long upper_difference();
};

#endif  // SBP_DIST_BLOCKMODEL_TRIPLET_HPP