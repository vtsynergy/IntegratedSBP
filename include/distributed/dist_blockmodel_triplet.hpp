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