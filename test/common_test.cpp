#include <vector>

#include <gtest/gtest.h>

#include "common.hpp"

#include "toy_example.hpp"
#include "utils.hpp"
#include "typedefs.hpp"

class CommonTest : public ToyExample {
};

class CommonComplexTest : public ComplexExample {
};

TEST_F(CommonTest, NewBlockDegreesAreCorrectlyComputed) {
    long vertex = 7;
    long current_block = B.block_assignment(vertex);
    utils::ProposalAndEdgeCounts proposal {0, 2, 3, 5};
    long current_block_self_edges = 3;
    long proposed_block_self_edges = 8;
    common::NewBlockDegrees nbd = common::compute_new_block_degrees(
            current_block, B, current_block_self_edges, proposed_block_self_edges, proposal);
    EXPECT_EQ(nbd.block_degrees_out[0], 10);
    EXPECT_EQ(nbd.block_degrees_out[1], 7);
    EXPECT_EQ(nbd.block_degrees_out[2], 6);
    EXPECT_EQ(nbd.block_degrees_in[0], 12);
    EXPECT_EQ(nbd.block_degrees_in[1], 7);
    EXPECT_EQ(nbd.block_degrees_in[2], 4);
    // TODO: when computing new_block_degrees, fix error where _block_degrees are improperly calculated (k != k_in + k_out if there are self_edges)
    // TODO: same error may be present in blockmodel creation and updates
    EXPECT_EQ(nbd.block_degrees[0], 14);
    EXPECT_EQ(nbd.block_degrees[1], 9);
    EXPECT_EQ(nbd.block_degrees[2], 7);
}

TEST_F(CommonComplexTest, NewBlockDegreesAreCorrectlyComputedWithBlockmodelDeltas) {
    long current_block = 3;
    long current_block_self_edges = B.blockmatrix()->get(current_block, current_block)
                                   + Deltas.get(current_block, current_block);
    long proposed_block_self_edges = B.blockmatrix()->get(Proposal.proposal, Proposal.proposal)
                                    + Deltas.get(Proposal.proposal, Proposal.proposal);
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
            current_block, B, current_block_self_edges, proposed_block_self_edges, Proposal);
    for (long block = 0; block < 6; ++block) {
        EXPECT_EQ(new_block_degrees.block_degrees_out[block], BlockDegreesAfterUpdates.block_degrees_out[block]);
        EXPECT_EQ(new_block_degrees.block_degrees_in[block], BlockDegreesAfterUpdates.block_degrees_in[block]);
        EXPECT_EQ(new_block_degrees.block_degrees[block], BlockDegreesAfterUpdates.block_degrees[block]);
    }
}

TEST_F(CommonComplexTest, NewBlockDegreesAreCorrectlyComputedWithDenseEdgeCountUpdates) {
    long current_block = 3;
    long current_block_self_edges = Updates.block_row[current_block];
    long proposed_block_self_edges = Updates.proposal_row[Proposal.proposal];
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
            current_block, B, current_block_self_edges, proposed_block_self_edges, Proposal);
    for (long block = 0; block < 6; ++block) {
        EXPECT_EQ(new_block_degrees.block_degrees_out[block], BlockDegreesAfterUpdates.block_degrees_out[block]);
        EXPECT_EQ(new_block_degrees.block_degrees_in[block], BlockDegreesAfterUpdates.block_degrees_in[block]);
        EXPECT_EQ(new_block_degrees.block_degrees[block], BlockDegreesAfterUpdates.block_degrees[block]);
    }
}

TEST_F(CommonComplexTest, NewBlockDegreesAreCorrectlyComputedWithSparseEdgeCountUpdates) {
    long current_block = 3;
    long current_block_self_edges = SparseUpdates.block_row[current_block];
    long proposed_block_self_edges = SparseUpdates.proposal_row[Proposal.proposal];
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
            current_block, B, current_block_self_edges, proposed_block_self_edges, Proposal);
    for (long block = 0; block < 6; ++block) {
        EXPECT_EQ(new_block_degrees.block_degrees_out[block], BlockDegreesAfterUpdates.block_degrees_out[block]);
        EXPECT_EQ(new_block_degrees.block_degrees_in[block], BlockDegreesAfterUpdates.block_degrees_in[block]);
        EXPECT_EQ(new_block_degrees.block_degrees[block], BlockDegreesAfterUpdates.block_degrees[block]);
    }
}

// TODO: new test to make sure proposal has the correct value for `num_neighbor_edges`
