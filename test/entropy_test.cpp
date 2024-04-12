#include <vector>

#include <gtest/gtest.h>
//#include <gmock/gmock.h>

#include "blockmodel.hpp"
#include "blockmodel/sparse/delta.hpp"
#include "entropy.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "utils.hpp"

#include "toy_example.hpp"
#include "typedefs.hpp"

// TODO: figure out correct placement of these
//MPI_t mpi;  // Unused
//Args args;  // Unused

const double MDL_10_VERTICES_50_EDGES_V1 = 200.5231073;
const double MDL_10_VERTICES_50_EDGES_V2 = 314.1041264;

const long E = 5E12;
const long V = 1E12;

class MockGraph : public Graph {
public:
    long num_edges() const override { return E; }
};

class EntropyTest : public ToyExample {
protected:
    Blockmodel B3;

    void SetUp() override {
        ToyExample::SetUp();
        std::vector<long> assignment3 = {0, 0, 0, 1, 2, 3, 3, 4, 5, 1, 5};
        B3 = Blockmodel(6, graph, 0.5, assignment3);
//        Deltas[std::make_pair(0, 0)] = 1;
//        Deltas[std::make_pair(0, 1)] = 0;
//        Deltas[std::make_pair(0, 2)] = 1;
//        Deltas[std::make_pair(1, 0)] = 1;
//        Deltas[std::make_pair(1, 2)] = -1;
//        Deltas[std::make_pair(2, 0)] = 1;
//        Deltas[std::make_pair(2, 1)] = 0;
//        Deltas[std::make_pair(2, 2)] = -3;
    }
};

class BlockMergeEntropyTest : public BlockMergeTest {};

TEST_F(EntropyTest, SetUpWorksCorrectly) {
    EXPECT_EQ(graph.num_vertices(), 11);
    EXPECT_EQ(graph.out_neighbors().size(), graph.num_vertices());
    EXPECT_EQ(graph.out_neighbors().size(), graph.in_neighbors().size());
    EXPECT_EQ(graph.num_edges(), 23);
}

TEST_F(EntropyTest, MDLGivesCorrectAnswer) {
    double E = entropy::mdl(B, graph);  // graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(E, ENTROPY) << "Calculated entropy = " << E << " but was expecting " << ENTROPY;
}

/// TODO: same test but using a vertex with a self edge
TEST_F(EntropyTest, DenseDeltaMDLGivesCorrectAnswer) {
    long vertex = 7;
    double E_before = entropy::mdl(B, graph);  // graph.num_vertices(), graph.num_edges());
    long current_block = B.block_assignment(vertex);
    double delta_entropy =
            entropy::delta_mdl(current_block, Proposal.proposal, B, graph.num_edges(), Updates, new_block_degrees);
    std::cout << "dE using updates = " << delta_entropy;
    B.move_vertex(V7, current_block, Proposal.proposal, Updates, new_block_degrees.block_degrees_out,
                  new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    double E_after = entropy::mdl(B, graph);  // graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(delta_entropy, E_after - E_before)
                        << "calculated dE was " << delta_entropy << " but actual dE was " << E_after - E_before;
}

TEST_F(EntropyTest, SparseDeltaMDLGivesCorrectAnswer) {
    long vertex = 7;
    double E_before = entropy::mdl(B, graph);  // graph.num_vertices(), graph.num_edges());
    long current_block = B.block_assignment(vertex);
    double delta_entropy =
            entropy::delta_mdl(current_block, Proposal.proposal, B, graph.num_edges(), SparseUpdates,
                               new_block_degrees);
    std::cout << "dE using sparse updates = " << delta_entropy;
    B.move_vertex(V7, current_block, Proposal.proposal, Updates, new_block_degrees.block_degrees_out,
                  new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    double E_after = entropy::mdl(B, graph);  // .num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(delta_entropy, E_after - E_before)
                        << "calculated dE was " << delta_entropy << " but actual dE was " << E_after - E_before;
}

/// TODO: same test but using a vertex with a self edge
TEST_F(EntropyTest, DeltaMDLUsingBlockmodelDeltasGivesCorrectAnswer) {
    long vertex = 7;
    double E_before = entropy::mdl(B, graph);  // .num_vertices(), graph.num_edges());
    double delta_entropy = entropy::delta_mdl(B, Deltas, Proposal);
    B.move_vertex(V7, Deltas, Proposal);
    long blockmodel_edges = utils::sum<long>(B.blockmatrix()->values());
    EXPECT_EQ(blockmodel_edges, graph.num_edges())
                        << "edges in blockmodel = " << blockmodel_edges << " edges in graph = " << graph.num_edges();
    double E_after = entropy::mdl(B, graph);  // .num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(delta_entropy, E_after - E_before) << "calculated dE was " << delta_entropy
                                                       << " but actual dE was " << E_after << " - " << E_before << " = "
                                                       << E_after - E_before;
}

TEST_F(EntropyTest, HastingsCorrectionBlockCountsAreTheSameWithAndWithoutBlockmodelDeltas) {
    long vertex = 7;
    MapVector<long> block_counts1;
//    std::unordered_map<long, long> block_counts1;
    for (const long neighbor: graph.out_neighbors(vertex)) {
        long neighbor_block = B.block_assignment(neighbor);
        block_counts1[neighbor_block] += 1;
    }
    for (const long neighbor: graph.in_neighbors(vertex)) {
        if (neighbor == vertex) continue;
        long neighbor_block = B.block_assignment(neighbor);
        block_counts1[neighbor_block] += 1;
    }
    utils::print(block_counts1);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex);
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    MapVector<long> block_counts2;
//    std::unordered_map<long, long> block_counts2;
    for (ulong i = 0; i < blocks_out_neighbors.indices.size(); ++i) {
        long block = blocks_out_neighbors.indices[i];
        long weight = blocks_out_neighbors.values[i];
        block_counts2[block] += weight; // block_count[new block] should initialize to 0
    }
    for (ulong i = 0; i < blocks_in_neighbors.indices.size(); ++i) {
        long block = blocks_in_neighbors.indices[i];
        long weight = blocks_in_neighbors.values[i];
        block_counts2[block] += weight; // block_count[new block] should initialize to 0
    }
    utils::print(block_counts2);
    for (const auto entry: block_counts1) {
        EXPECT_EQ(entry.second, block_counts2[entry.first]);
    }
    for (const auto entry: block_counts2) {
        EXPECT_EQ(entry.second, block_counts1[entry.first]);
    }
}

TEST_F(EntropyTest, HastingsCorrectionWithAndWithoutDeltaGivesSameResult) {
    long vertex = 7;
    long current_block = B.block_assignment(vertex);
    double hastings1 = entropy::hastings_correction(vertex, graph, B, Deltas, current_block, Proposal);
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex);
    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
    double hastings2 = entropy::hastings_correction(B, blocks_out_neighbors, blocks_in_neighbors, Proposal, Updates,
                                                    new_block_degrees);
    EXPECT_FLOAT_EQ(hastings1, hastings2);
}

TEST_F(EntropyTest, SpecialCaseShouldGiveCorrectDeltaMDL) {
    long vertex = 6;
    utils::ProposalAndEdgeCounts proposal{0, 1, 2, 3};
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, true);
    SparseEdgeCountUpdates updates;
    finetune::edge_count_updates_sparse(B3, vertex, 3, 0, out_edges, in_edges, updates);
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
            3, B3, 1, 4, proposal);
    std::cout << "before copies" << std::endl;
    Blockmodel B4 = B3.copy();
    Blockmodel B5 = B3.copy();
    std::cout << "before move_vertex" << std::endl;
    VertexMove result = finetune::move_vertex(6, 3, proposal, B4, graph, out_edges, in_edges);
    std::cout << "before blockmodel.move_vertex" << std::endl;
    B5.move_vertex(V6, 3, 0, updates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in,
                   new_block_degrees.block_degrees);
    std::cout << "before mdl" << std::endl;
    double E_before = entropy::mdl(B3, graph);  // 11, 23);
    double dE = entropy::mdl(B5, graph) - E_before;  // 11, 23) - E_before;
    std::cout << "======== Before move ========" << std::endl;
    B3.print_blockmodel();
    std::cout << "======== After move =======" << std::endl;
    B5.print_blockmodel();
    EXPECT_FLOAT_EQ(dE, result.delta_entropy);
}

//TEST_F(EntropyTest, NullModelMDLv1ShouldGiveCorrectMDLForSmallGraph) {
//    double mdl = entropy::null_mdl_v1(50);
//    EXPECT_FLOAT_EQ(mdl, MDL_10_VERTICES_50_EDGES_V1);
//}

TEST_F(EntropyTest, NullModelMDLv2ShouldGiveCorrectMDLForSmallGraph) {
    double mdl = entropy::null_mdl_v2(10, 50);
    EXPECT_FLOAT_EQ(mdl, MDL_10_VERTICES_50_EDGES_V2);
}

TEST_F(EntropyTest, NullModelMDLv1ShouldGiveCorrectMDLForLargeGraph) {
    double hand_calculated_mdl = 1.462023E14;
    double blocks = 1.0;
    double x = (blocks * blocks) / double(E);
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    double bm = (double(E) * h) + double(V) * log(blocks);
    double log_likelihood_p = double(E) * log(double(E) / (double(E) * double(E)));
    double result = bm - log_likelihood_p;
//    Graph largeGraph;
    MockGraph LargeGraph;
//    EXPECT_CALL(LargeGraph, num_edges()).Times(::testing::AtLeast(4));
//    std::cout << "E: " << LargeGraph.num_edges() << std::endl;
//    EXPECT_CALL(LargeGraph, num_edges()).WillRepeatedly(::testing::Return(E));
//    EXPECT_EQ(LargeGraph.num_edges(), E);
//    largeGraph.num_vertices = V;
//    largeGraph.num_edges = E;
    double mdl = entropy::null_mdl_v1(LargeGraph);
    EXPECT_FLOAT_EQ(mdl, result);
    EXPECT_FLOAT_EQ(mdl, hand_calculated_mdl);
}

TEST_F(EntropyTest, NullModelMDLv2ShouldGiveCorrectMDLForLargeGraph) {
    double hand_calculated_mdl = 3.089407E+14;
    long E = 5E12;
    long V = 1E12;
    auto blocks = double(V);
    double x = (blocks * blocks) / double(E);
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    double bm = (double(E) * h) + double(V) * log(blocks);
    double cell_value = double(E) / (blocks * blocks);
    double cell_degree = double(E) / blocks;
    double log_likelihood_p = (blocks * blocks) * (cell_value * log(cell_value / (cell_degree * cell_degree)));
    double result = bm - log_likelihood_p;
    double mdl = entropy::null_mdl_v2(V, E);
    EXPECT_FLOAT_EQ(mdl, result);
    EXPECT_FLOAT_EQ(mdl, hand_calculated_mdl);
}

TEST_F(BlockMergeEntropyTest, BlockmodelDeltaMDLIsCorrectlyComputeWithDenseUpdates) {
    double E_before = entropy::mdl(B, graph);  // 11, 23);
    double dE = entropy::block_merge_delta_mdl(0, 1, 23, B, Updates, new_block_degrees);
    double E_after = entropy::mdl(B2, graph);  // 11, 23);
    EXPECT_FLOAT_EQ(E_after - E_before, dE);
}

TEST_F(BlockMergeEntropyTest, BlockmodelDeltaMDLIsCorrectlyComputeWithSparseUpdates) {
    double E_before = entropy::mdl(B, graph);  // 11, 23);
    double dE = entropy::block_merge_delta_mdl(0, 1, 23, B, SparseUpdates, new_block_degrees);
    double E_after = entropy::mdl(B2, graph);  // 11, 23);
    EXPECT_FLOAT_EQ(E_after - E_before, dE);
}

TEST_F(BlockMergeEntropyTest, BlockmodelDeltaMDLIsCorrectlyComputeWithBlockmodelDeltas) {
    double E_before = entropy::mdl(B, graph);  // 11, 23);
    double dE = entropy::block_merge_delta_mdl(0, B, Deltas, new_block_degrees);
    double E_after = entropy::mdl(B2, graph);  // 11, 23);
    EXPECT_FLOAT_EQ(E_after - E_before, dE);
}

TEST_F(BlockMergeEntropyTest, BlockmodelDeltaMDLIsCorrectlyComputeWithBlockmodelDeltasSansBlockDegrees) {
    double E_before = entropy::mdl(B, graph);  // 11, 23);
    double dE = entropy::block_merge_delta_mdl(0, {1, B.degrees_out(0),
                                                       B.degrees_in(0), B.degrees(0)}, B, Deltas);
    double E_after = entropy::mdl(B2, graph);  // 11, 23);
    EXPECT_FLOAT_EQ(E_after - E_before, dE);
}
