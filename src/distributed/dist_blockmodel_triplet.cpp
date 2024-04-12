#include "distributed/dist_blockmodel_triplet.hpp"

TwoHopBlockmodel DistBlockmodelTriplet::get_next_blockmodel(TwoHopBlockmodel &old_blockmodel) {
    if (mpi.rank == 0) std::cout << "2 hop getting next blockmodel" << std::endl;
    old_blockmodel.setNum_blocks_to_merge(0);
    this->update(old_blockmodel);
    this->status();
    // If have not yet reached golden ratio, continue from middle blockmodel
    if (this->golden_ratio_not_reached()) {
        if (mpi.rank == 0) std::cout << "golden ratio NOT reached" << std::endl;
        if (mpi.rank == 0) std::cout << std::boolalpha << this->get(0).empty << " " << this->get(1).empty << " " << this->get(2).empty << std::endl;
        TwoHopBlockmodel blockmodel = this->get(1).copy();
        blockmodel.setNum_blocks_to_merge(long(blockmodel.getNum_blocks() * BLOCK_REDUCTION_RATE));
        if (blockmodel.getNum_blocks_to_merge() == 0) {
            this->optimal_num_blocks_found = true;
        }
        return blockmodel;
    }
    // If community detection is done, return the middle (optimal) blockmodel
    if (this->is_done()) {
        return this->get(1).copy();
    }
    // Find which Blockmodel would serve as the starting point for the next iteration
    long index = 1;
    // TODO: if things get funky, look into this if/else statement
    if (this->get(0).empty && this->get(1).getNum_blocks() > this->get(2).getNum_blocks()) {
        index = 1;
    } else if (this->upper_difference() >= this->lower_difference()) {
        index = 0;
    } else {
        index = 1;
    }
    long next_num_blocks_to_try = this->get(index + 1).getNum_blocks();
    next_num_blocks_to_try += long((this->get(index).getNum_blocks() - this->get(index + 1).getNum_blocks()) * 0.618);
    TwoHopBlockmodel blockmodel = this->get(index).copy();
    blockmodel.setNum_blocks_to_merge(blockmodel.getNum_blocks() - next_num_blocks_to_try);
    return blockmodel;
}

bool DistBlockmodelTriplet::golden_ratio_not_reached() { return this->get(2).empty; }

bool DistBlockmodelTriplet::is_done() {
    if ((!this->get(0).empty && this->get(0).getNum_blocks() - this->get(2).getNum_blocks() == 2) ||
        (this->get(0).empty && this->get(1).getNum_blocks() - this->get(2).getNum_blocks() == 1)) {
        this->optimal_num_blocks_found = true;
    }
    return this->optimal_num_blocks_found;
}

long DistBlockmodelTriplet::lower_difference() {
    return this->get(1).getNum_blocks() - this->get(2).getNum_blocks();
}

long DistBlockmodelTriplet::upper_difference() {
    return this->get(0).getNum_blocks() - this->get(1).getNum_blocks();
}

void DistBlockmodelTriplet::status() {
    double entropies[3];
    long num_blocks[3];
    for (long i = 0; i < 3; ++i) {
        if (this->blockmodels[i].empty) {
            entropies[i] = std::numeric_limits<double>::min();
            num_blocks[i] = 0;
        } else {
            entropies[i] = this->blockmodels[i].getOverall_entropy();
            num_blocks[i] = this->blockmodels[i].getNum_blocks();
        }
    }
    if (mpi.rank == 0) std::cout << "Overall entropy: " << entropies[0] << " " << entropies[1] << " " << entropies[2] << std::endl;
    if (mpi.rank == 0) std::cout << "Number of blocks: " << num_blocks[0] << " " << num_blocks[1] << " " << num_blocks[2] << std::endl;
    if (this->optimal_num_blocks_found) {
        if (mpi.rank == 0) std::cout << "Optimal blockmodel found with " << num_blocks[1] << " blocks" << std::endl;
    } else if (!(this->golden_ratio_not_reached())) {
        if (mpi.rank == 0) std::cout << "Golden ratio has been reached" << std::endl;
    }
}

void DistBlockmodelTriplet::update(TwoHopBlockmodel &blockmodel) {
    if (mpi.rank == 0) std::cout << "updating triplet with BM with B = " << blockmodel.getNum_blocks() << std::endl;
    long index;
    if (this->blockmodels[1].empty) {
        index = 1;
    } else {
        if (blockmodel.getOverall_entropy() <= this->blockmodels[1].getOverall_entropy()) {
            long old_index;
            if (this->get(1).getNum_blocks() > blockmodel.getNum_blocks()) {
                old_index = 0;
            } else {
                old_index = 2;
            }
            this->blockmodels[old_index] = this->blockmodels[1];
            index = 1;
        } else {
            if (this->blockmodels[1].getNum_blocks() > blockmodel.getNum_blocks()) {
                index = 2;
            } else {
                index = 0;
            }
        }
    }
    if (mpi.rank == 0) std::cout << "found the index to be = " << index << std::endl;
    this->blockmodels[index] = blockmodel;
    if (mpi.rank == 0) std::cout << "blockmodels[index] has B = " << this->blockmodels[index].getNum_blocks() << std::endl;
}