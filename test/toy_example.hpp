#ifndef DISTRIBUTEDSBP_TEST_TOY_EXAMPLE_H
#define DISTRIBUTEDSBP_TEST_TOY_EXAMPLE_H

#include <vector>

#include <gtest/gtest.h>

#include "args.hpp"
#include "blockmodel.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "mpi_data.hpp"
#include "rng.hpp"
#include "utils.hpp"
#include "typedefs.hpp"

const double ENTROPY = 92.58797747;

class ToyExample : public ::testing::Test {
protected:
    // My variables
    std::vector<long> assignment;
    std::vector<bool> self_edges;
    Blockmodel B, B2, B3;
    utils::ProposalAndEdgeCounts Proposal;
    Graph graph;
    common::NewBlockDegrees  new_block_degrees;
    EdgeCountUpdates Updates;
    SparseEdgeCountUpdates SparseUpdates;
    Delta Deltas;
    VertexMove_v3 Move;
    VertexMove_v3 SelfEdgeMove;
    Vertex V5, V6, V7;

    void SetUp() override {
        ToySetUp(true);
    }

    void ToySetUp(bool transpose) {
        args.threads = 1;
        rng::init_generators();
        args.transpose = transpose;
        std::vector<std::vector<long>> edges {
                {0, 0},
                {0, 1},
                {0, 2},
                {1, 2},
                {2, 3},
                {3, 1},
                {3, 2},
                {3, 5},
                {4, 1},
                {4, 6},
                {5, 4},
                {5, 5},
                {5, 6},
                {5, 7},
                {6, 4},
                {7, 3},
                {7, 9},
                {8, 5},
                {8, 7},
                {9, 10},
                {10, 7},
                {10, 8},
                {10, 10}
        };
        V5 = { 5, 4, 3 };
        V6 = { 6, 1, 2 };
        V7 = { 7, 2, 3 };
        long num_vertices = 11;
        long num_edges = (long) edges.size();
        assignment = { 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2 };
        self_edges = { true, false, false, false, false, true, false, false, false, false, true };
        NeighborList out_neighbors;
        NeighborList in_neighbors;
        for (const std::vector<long> &edge : edges) {
            long from = edge[0];
            long to = edge[1];
            utils::insert_nodup(out_neighbors, from , to);
            utils::insert_nodup(in_neighbors, to, from);
        }
        graph = Graph(out_neighbors, in_neighbors, num_vertices, num_edges, self_edges, assignment);
        B = Blockmodel(3, graph, 0.5, assignment);
        new_block_degrees.block_degrees_out = { 10, 7, 6 };
        new_block_degrees.block_degrees_in = { 12, 7, 4 };
        new_block_degrees.block_degrees = { 14, 9, 7 };
        Updates.block_row = { 2, 1, 3 };
        Updates.block_col = { 1, 0, 3 };
        Updates.proposal_row = { 8, 1, 1 };
        Updates.proposal_col = { 8, 2, 2 };
        SparseUpdates.block_row[0] = 2;
        SparseUpdates.block_row[1] = 1;
        SparseUpdates.block_row[2] = 3;
        SparseUpdates.block_col[0] = 1;
        SparseUpdates.block_col[2] = 3;
        SparseUpdates.proposal_row[0] = 8;
        SparseUpdates.proposal_row[1] = 1;
        SparseUpdates.proposal_row[2] = 1;
        SparseUpdates.proposal_col[0] = 8;
        SparseUpdates.proposal_col[1] = 2;
        SparseUpdates.proposal_col[2] = 2;
        Deltas = Delta(2, 0);
        Deltas.add(0, 0, 1);
        Deltas.add(0, 2, 1);
        Deltas.add(1, 0, 1);
        Deltas.add(1, 2, -1);
        Deltas.add(2, 0, 1);
        Deltas.add(2, 2, -3);
        Proposal = {0, 2, 3, 5};
        std::vector<long> assignment2(assignment);
        assignment2[7] = Proposal.proposal;
        B2 = Blockmodel(3, graph, 0.5, assignment2);
        Move = {
                -0.01,  // random change in entropy value
                true,
                V7,
                Proposal.proposal,
                EdgeWeights { { 3, 9}, { 1, 1 } },
                EdgeWeights { { 5, 8, 10 }, { 1, 1, 1 }}
        };
        SelfEdgeMove = {
                -0.01,
                true,
                V5,
                0,
                EdgeWeights { { 4, 5, 6, 7 }, { 1, 1, 1, 1 } },
                EdgeWeights { { 3, 8 }, { 1, 1 }}
        };
        std::vector<long> assignment3(assignment);
        assignment3[5] = 0;
        B3 = Blockmodel(3, graph, 0.5, assignment3);
    }
//    virtual void TearDown() {
//
//    }
};

class BlockMergeTest : public ToyExample {
    void SetUp() override {
        ToyExample::SetUp();
        ToySetUp(true);
        Updates = EdgeCountUpdates();
        Updates.block_row = { 0, 0, 0 };
        Updates.block_col = { 0, 0, 0 };
        Updates.proposal_row = { 0, 14, 1 };
        Updates.proposal_col = { 0, 14, 2 };
        SparseUpdates = SparseEdgeCountUpdates();
        SparseUpdates.proposal_row[1] = 14;
        SparseUpdates.proposal_row[2] = 1;
        SparseUpdates.proposal_col[1] = 14;
        SparseUpdates.proposal_col[2] = 2;
        Deltas = Delta(0, 1);
        Deltas.add(0, 0, -7);
        Deltas.add(0, 1, -1);
        Deltas.add(1, 0, -1);
        Deltas.add(1, 1, 9);
        Deltas.add(2, 0, -1);
        Deltas.add(2, 1, 1);
        new_block_degrees.block_degrees_out = { 0, 15, 8 };
        new_block_degrees.block_degrees_in = { 0, 16, 7 };
        new_block_degrees.block_degrees = { 0, 17, 9 };
        std::vector<long> assignment2 = { 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2 };
        B2 = Blockmodel(3, graph, 0.5, assignment2);
    }
};

class ComplexExample : public ToyExample {
protected:
    common::NewBlockDegrees BlockDegreesAfterUpdates;

    void SetUp() override {
        ToyExample::SetUp();
        ToyExample::ToySetUp(true);
        ComplexToySetUp(true);
    }

    void ComplexToySetUp(bool transpose) {
        args.transpose = transpose;
        Proposal = { 0, 1, 2, 3 };
        assignment = { 0, 0, 0, 1, 2, 3, 3, 4, 5, 1, 5 };
        B = Blockmodel(6, graph, 0.5, assignment);
        std::vector<long> assignment2(assignment);
        assignment2[6] = Proposal.proposal;
        B2 = Blockmodel(6, graph, 0.5, assignment2);
        Updates.block_row = { 1, 0, 1, 1, 1, 0 };
        Updates.block_col = { 0, 1, 0, 1, 0, 1 };
        Updates.proposal_row = { 4, 1, 1, 0, 0, 0 };
        Updates.proposal_col = { 4, 2, 2, 1, 0, 0 };
        SparseUpdates = SparseEdgeCountUpdates();
        SparseUpdates.block_row[0] = 1;
        SparseUpdates.block_row[2] = 1;
        SparseUpdates.block_row[3] = 1;
        SparseUpdates.block_row[4] = 1;
        SparseUpdates.block_col[1] = 1;
        SparseUpdates.block_col[3] = 1;
        SparseUpdates.block_col[5] = 1;
        SparseUpdates.proposal_row[0] = 4;
        SparseUpdates.proposal_row[1] = 1;
        SparseUpdates.proposal_row[2] = 1;
        SparseUpdates.proposal_col[0] = 4;
        SparseUpdates.proposal_col[1] = 2;
        SparseUpdates.proposal_col[2] = 2;
        SparseUpdates.proposal_col[3] = 1;
        Deltas = Delta(3, 0);
        Deltas.add(3, 0, 1);
        Deltas.add(3, 2, -1);
        Deltas.add(3, 3, -1);
        Deltas.add(2, 3, -1);
        Deltas.add(0, 2, 1);
        Deltas.add(2, 0, 1);
        new_block_degrees.block_degrees_out = { 7, 3, 1, 4, 0, 1 };
        new_block_degrees.block_degrees_in = { 8, 1, 2, 4, 1, 0 };
        new_block_degrees.block_degrees = { 11, 4, 3, 7, 1, 1 };
        BlockDegreesAfterUpdates.block_degrees_out = { 6, 4, 2, 4, 2, 5 };
        BlockDegreesAfterUpdates.block_degrees_in = { 9, 3, 2, 3, 3, 3 };
        BlockDegreesAfterUpdates.block_degrees = { 11, 7, 4, 6, 5, 6 };
        V6 = { 6, 1, 2 };
        Move = {
                -0.01,  // random change in entropy value
                true,
                V6,
                Proposal.proposal,
                EdgeWeights { { 4 }, { 1 } },
                EdgeWeights { { 4, 5 }, { 1, 1 }}
        };
        std::vector<long> assignment3(assignment);
        assignment3[5] = Proposal.proposal;
        B3 = Blockmodel(6, graph, 0.5, assignment3);
    }
//    virtual void TearDown() {
//
//    }
};

static void VECTOR_EQ(const std::vector<long> &a, const std::vector<long> &b) {
    EXPECT_EQ(a.size(), b.size());
    for (int i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i], b[i]) << "vectors differ at index " << i << " | " << a[i] << " /= " << b[i];
    }
}

static void IS_SORTED(const std::vector<long> &a) {
    for (int i = 0; i < a.size() - 1; ++i) {
        EXPECT_TRUE(a[i] >= a[i + 1]) << "vector is not sorted: " << a[i] << " !>= " << a[i + 1];
    }
}

#endif //DISTRIBUTEDSBP_TEST_TOY_EXAMPLE_H
