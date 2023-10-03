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
 * Stores the triplet of blockmodels needed for the fibonacci search.
 */
#ifndef CPPSBP_PARTITION_PARTITION_TRIPLET_HPP
#define CPPSBP_PARTITION_PARTITION_TRIPLET_HPP

#include <iostream>
#include <limits>

#include "blockmodel.hpp"

class BlockmodelTriplet {

public:
    /// Creates an empty blockmodel triplet.
    BlockmodelTriplet() : optimal_num_blocks_found(false) {}
    /// True if the optimal number of blocks has been found.
    bool optimal_num_blocks_found;
    /// Updates the blockmodel triplet with the provided blockmodel (inserts it into the correct spot, moves/deletes
    /// the remaining blockmodels as needed).
    void update(Blockmodel &blockmodel);
    /// prints the number of blocks and overall entropy of every blockmodel in the blockmodel triplet.
    void status();
    /// Returns the blockmodel at index `i`. IndexOutOfBoundError if i >= 3.
    Blockmodel &get(long i) { return blockmodels[i]; }
    /// Returns true if the golden ratio has not been reached, false otherwise. In practice, returns true if
    /// blockmodels[2] == 0.
    bool golden_ratio_not_reached();
    /// Returns true if the optimal number of blocks has been found.
    bool is_done();
    /// Returns the blockmodel on which to perform the next iteration of SBP.
    Blockmodel get_next_blockmodel(Blockmodel &old_blockmodel);

protected:
    /// Blockmodels arranged in order of decreasing number of blocks.
    /// If the first blockmodel is empty, then the golden ratio bracket has not yet been established.
    Blockmodel blockmodels[3];
    /// TODO
    long lower_difference();
    /// TODO
    long upper_difference();
};

#endif // CPPSBP_PARTITION_PARTITION_TRIPLET_HPP
