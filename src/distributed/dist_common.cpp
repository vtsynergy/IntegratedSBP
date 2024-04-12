#include "distributed/dist_common.hpp"

#include "common.hpp"

namespace common::dist {

// TODO: get rid of block_assignment, just use blockmodel?
utils::ProposalAndEdgeCounts propose_new_block(long current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                               const std::vector<long> &block_assignment,
                                               const TwoHopBlockmodel &blockmodel, bool block_merge) {
    std::vector<long> neighbor_indices = utils::concatenate<long>(out_blocks.indices, in_blocks.indices);
    std::vector<long> neighbor_weights = utils::concatenate<long>(out_blocks.values, in_blocks.values);
    long k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
    long k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
    long k = k_out + k_in;
    long num_blocks = blockmodel.getNum_blocks();

    if (k == 0) { // If the current block has no neighbors, propose merge with random block
        std::vector<long> blocks = utils::range<long>(0, blockmodel.getNum_blocks());
        std::vector<long> weights = utils::to_long<bool>(blockmodel.in_two_hop_radius());
        long proposal = choose_neighbor(blocks, weights);
        // long proposal = propose_random_block(current_block, num_blocks);  // TODO: only propose blocks in 2 hop radius
        assert(blockmodel.stores(proposal));
        return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }
    long neighbor_block;
    if (block_merge)
        neighbor_block = choose_neighbor(neighbor_indices, neighbor_weights);
    else {
        long neighbor = choose_neighbor(neighbor_indices, neighbor_weights);
        neighbor_block = block_assignment[neighbor];
    }
    assert(blockmodel.stores(neighbor_block));

    // With a probability inversely proportional to block degree, propose a random block merge
    if (std::rand() <= (num_blocks / ((double) blockmodel.degrees(neighbor_block) + num_blocks))) {
        std::vector<long> blocks = utils::range<long>(0, blockmodel.getNum_blocks());
        std::vector<long> weights = utils::to_long<bool>(blockmodel.in_two_hop_radius());
        long proposal = choose_neighbor(blocks, weights);
        // long proposal = propose_random_block(current_block, num_blocks);
        assert(blockmodel.stores(proposal));
        return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }

    // Build multinomial distribution
    double total_edges = 0.0;
    MapVector<long> edges = blockmodel.blockmatrix()->neighbors_weights(neighbor_block);
    if (block_merge) {  // Make sure proposal != current_block
        edges[current_block] = 0;
        total_edges = utils::sum<double, long>(edges);
        if (total_edges == 0.0) { // Neighbor block has no neighbors, so propose a random block
            long proposal = propose_random_block(current_block, num_blocks);
            assert(blockmodel.stores(proposal));
            return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
        }
    } else {
        total_edges = utils::sum<double, long>(edges);
    }
    if (edges.empty()) {
        std::cerr << "ERROR " << "ERROR: NO EDGES! k = " << blockmodel.degrees(neighbor_block) << " "
        << blockmodel.degrees_out(neighbor_block) << " " << blockmodel.degrees_in(neighbor_block)
        << std::endl;
        utils::print<long>(blockmodel.blockmatrix()->getrow_sparse(neighbor_block));
        utils::print<long>(blockmodel.blockmatrix()->getcol_sparse(neighbor_block));
    }
    // Propose a block based on the multinomial distribution of block neighbor edges
    SparseVector<double> multinomial_distribution;
    utils::div(edges, total_edges, multinomial_distribution);
    long proposal = choose_neighbor(multinomial_distribution);
    assert(blockmodel.stores(proposal));
    return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
}

}  // namespace common::dist