#include <vector>

#include <gtest/gtest.h>

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
const double NONPARAMETRIC_ENTROPY = 69.30665809839104;
const double SPARSE_ENTROPY = 19.122212674046708;
const double SIMPLE_NONPARAMETRIC_ENTROPY = 71.29003084265679;
const double SIMPLE_SPARSE_ENTROPY = 39.85008728133248;
const double PARTITION_DL = 15.558998478518633;
const double DEGREE_DL = 18.744501863020016;
const double EDGES_DL = 15.88094508280568;

class NonparametricEntropyTest : public ToyExample {
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

class NonparametricBlockMergeEntropyTest : public BlockMergeTest {};

TEST_F(NonparametricEntropyTest, SetUpWorksCorrectly) {
    EXPECT_EQ(graph.num_vertices(), 11);
    EXPECT_EQ(graph.out_neighbors().size(), graph.num_vertices());
    EXPECT_EQ(graph.out_neighbors().size(), graph.in_neighbors().size());
    EXPECT_EQ(graph.num_edges(), 23);
}

TEST_F(NonparametricEntropyTest, MDLGivesCorrectAnswer) {
    utils::print<long>(B.block_assignment());
    double E = entropy::nonparametric::mdl(B, graph);  // graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(E, SIMPLE_NONPARAMETRIC_ENTROPY) << "Calculated entropy = " << E << " but was expecting " << SIMPLE_NONPARAMETRIC_ENTROPY;
}

TEST_F(NonparametricEntropyTest, SparseEntropyGivesCorrectAnswer) {
    utils::print<long>(B.block_assignment());
    double E = entropy::nonparametric::sparse_entropy(B, graph);  // graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(E, SIMPLE_SPARSE_ENTROPY) << "Calculated entropy = " << E << " but was expecting " << SIMPLE_SPARSE_ENTROPY;
}

TEST_F(NonparametricEntropyTest, DegreeCorrectedMDLGivesCorrectAnswer) {
    args.degreecorrected = true;
    utils::print<long>(B.block_assignment());
    double E = entropy::nonparametric::mdl(B, graph);  // graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(E, ENTROPY) << "Calculated entropy = " << E << " but was expecting " << NONPARAMETRIC_ENTROPY;
    args.degreecorrected = false;
}

TEST_F(NonparametricEntropyTest, DegreeCorrectedSparseEntropyGivesCorrectAnswer) {
    args.degreecorrected = true;
    utils::print<long>(B.block_assignment());
    double E = entropy::nonparametric::sparse_entropy(B, graph);  // graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(E, SPARSE_ENTROPY) << "Calculated entropy = " << E << " but was expecting " << SPARSE_ENTROPY;
    args.degreecorrected = false;
}

TEST_F(NonparametricEntropyTest, PartitionDLGivesCorrectAnswer) {
    utils::print<long>(B.block_assignment());
    double E = entropy::nonparametric::get_partition_dl(graph.num_vertices(), B);  // graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(E, PARTITION_DL) << "Calculated entropy = " << E << " but was expecting " << PARTITION_DL;
}

TEST_F(NonparametricEntropyTest, DegreeDLGivesCorrectAnswer) {
    utils::print<long>(B.block_assignment());
    double E = entropy::nonparametric::get_deg_dl_dist(B);  // graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(E, 0.00) << "Calculated entropy = " << E << " but was expecting " << 0.00;
}

TEST_F(NonparametricEntropyTest, DegreeCorrectedDegreeDLGivesCorrectAnswer) {
    args.degreecorrected = true;
    utils::print<long>(B.block_assignment());
    double E = entropy::nonparametric::get_deg_dl_dist(B);  // graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(E, DEGREE_DL) << "Calculated entropy = " << E << " but was expecting " << DEGREE_DL;
    args.degreecorrected = false;
}

TEST_F(NonparametricEntropyTest, EdgesDLGivesCorrectAnswer) {
    utils::print<long>(B.block_assignment());
    double E = entropy::nonparametric::get_edges_dl(B.num_nonempty_blocks(), graph.num_edges());  // graph.num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(E, EDGES_DL) << "Calculated entropy = " << E << " but was expecting " << EDGES_DL;
}

/// TODO: same test but using a vertex with a self edge
TEST_F(NonparametricEntropyTest, DeltaMDLUsingBlockmodelDeltasGivesCorrectAnswer) {
    long vertex = 7;
    double E_before = entropy::nonparametric::mdl(B, graph);  // .num_vertices(), graph.num_edges());
    double delta_entropy = entropy::nonparametric::delta_mdl(B, graph, vertex, Deltas, Proposal);
    B.move_vertex(V7, Deltas, Proposal);
    long blockmodel_edges = utils::sum<long>(B.blockmatrix()->values());
    EXPECT_EQ(blockmodel_edges, graph.num_edges())
                        << "edges in blockmodel = " << blockmodel_edges << " edges in graph = " << graph.num_edges();
    double E_after = entropy::nonparametric::mdl(B, graph);  // .num_vertices(), graph.num_edges());
    EXPECT_FLOAT_EQ(delta_entropy, E_after - E_before) << "calculated dE was " << delta_entropy
                                                       << " but actual dE was " << E_after << " - " << E_before << " = "
                                                       << E_after - E_before;
}

//TEST_F(NonparametricEntropyTest, HastingsCorrectionBlockCountsAreTheSameWithAndWithoutBlockmodelDeltas) {
//    long vertex = 7;
//    MapVector<long> block_counts1;
////    std::unordered_map<long, long> block_counts1;
//    for (const long neighbor: graph.out_neighbors(vertex)) {
//        long neighbor_block = B.block_assignment(neighbor);
//        block_counts1[neighbor_block] += 1;
//    }
//    for (const long neighbor: graph.in_neighbors(vertex)) {
//        if (neighbor == vertex) continue;
//        long neighbor_block = B.block_assignment(neighbor);
//        block_counts1[neighbor_block] += 1;
//    }
//    utils::print(block_counts1);
//    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex);
//    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex);
//    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
//    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
//    MapVector<long> block_counts2;
////    std::unordered_map<long, long> block_counts2;
//    for (ulong i = 0; i < blocks_out_neighbors.indices.size(); ++i) {
//        long block = blocks_out_neighbors.indices[i];
//        long weight = blocks_out_neighbors.values[i];
//        block_counts2[block] += weight; // block_count[new block] should initialize to 0
//    }
//    for (ulong i = 0; i < blocks_in_neighbors.indices.size(); ++i) {
//        long block = blocks_in_neighbors.indices[i];
//        long weight = blocks_in_neighbors.values[i];
//        block_counts2[block] += weight; // block_count[new block] should initialize to 0
//    }
//    utils::print(block_counts2);
//    for (const auto entry: block_counts1) {
//        EXPECT_EQ(entry.second, block_counts2[entry.first]);
//    }
//    for (const auto entry: block_counts2) {
//        EXPECT_EQ(entry.second, block_counts1[entry.first]);
//    }
//}

//TEST_F(NonparametricEntropyTest, HastingsCorrectionWithAndWithoutDeltaGivesSameResult) {
//    long vertex = 7;
//    long current_block = B.block_assignment(vertex);
//    double hastings1 = entropy::hastings_correction(vertex, graph, B, Deltas, current_block, Proposal);
//    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex);
//    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex);
//    EdgeWeights blocks_out_neighbors = finetune::block_edge_weights(B.block_assignment(), out_edges);
//    EdgeWeights blocks_in_neighbors = finetune::block_edge_weights(B.block_assignment(), in_edges);
//    double hastings2 = entropy::hastings_correction(B, blocks_out_neighbors, blocks_in_neighbors, Proposal, Updates,
//                                                    new_block_degrees);
//    EXPECT_FLOAT_EQ(hastings1, hastings2);
//}

TEST_F(NonparametricEntropyTest, SpecialCaseShouldGiveCorrectDeltaMDL) {
    long vertex = 6;
    utils::ProposalAndEdgeCounts proposal{0, 1, 2, 3};
    EdgeWeights out_edges = finetune::edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = finetune::edge_weights(graph.in_neighbors(), vertex, true);
    SparseEdgeCountUpdates updates;
    Delta deltas = finetune::blockmodel_delta(6, 3, 0, out_edges, in_edges, B3);
//    finetune::edge_count_updates_sparse(B3, vertex, 3, 0, out_edges, in_edges, updates);
//    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
//            3, B3, 1, 4, proposal);
    std::cout << "before copies" << std::endl;
    Blockmodel B4 = B3.copy();
    utils::print<long>(B4.block_assignment());
    Blockmodel B5 = B3.copy();
    std::cout << "before move_vertex" << std::endl;
    args.nonparametric = true;
    VertexMove result = finetune::move_vertex(6, 3, proposal, B4, graph, out_edges, in_edges);
    std::cout << "============ B5.move_vertex()" << std::endl;
    B5.move_vertex(V6, deltas, proposal);
//    B5.move_vertex(V6, 3, 0, updates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in,
//                   new_block_degrees.block_degrees);
    std::cout << "before mdl" << std::endl;
    B3.print_blockmodel();
    double E_before = entropy::nonparametric::mdl(B3, graph);  // 11, 23);
    EXPECT_FLOAT_EQ(E_before, 81.44696150567646);
    B5.print_blockmodel();
    double E_after = entropy::nonparametric::mdl(B5, graph);
    // TODO: why isn't the assignment in B5 correct??
    EXPECT_FLOAT_EQ(E_after, 82.12655765285804);
    double dE = E_after - E_before;  // 11, 23) - E_before;
//    std::cout << "======== Before move ========" << std::endl;
//    B3.print_blockmodel();
//    std::cout << "======== After move =======" << std::endl;
//    B5.print_blockmodel();
    EXPECT_FLOAT_EQ(dE, result.delta_entropy);
    args.nonparametric = false;
}

//TEST_F(NonparametricEntropyTest, NullModelMDLv1ShouldGiveCorrectMDLForSmallGraph) {
//    double mdl = entropy::null_mdl_v1(50);
//    EXPECT_FLOAT_EQ(mdl, MDL_10_VERTICES_50_EDGES_V1);
//}

//TEST_F(NonparametricEntropyTest, NullModelMDLv2ShouldGiveCorrectMDLForSmallGraph) {
//    double mdl = entropy::null_mdl_v2(10, 50);
//    EXPECT_FLOAT_EQ(mdl, MDL_10_VERTICES_50_EDGES_V2);
//}

//TEST_F(NonparametricEntropyTest, NullModelMDLv1ShouldGiveCorrectMDLForLargeGraph) {
//    double hand_calculated_mdl = 1.462023E14;
//    long E = 5E12;
//    long V = 1E12;
//    double blocks = 1.0;
//    double x = (blocks * blocks) / double(E);
//    double h = ((1 + x) * log(1 + x)) - (x * log(x));
//    double bm = (double(E) * h) + double(V) * log(blocks);
//    double log_likelihood_p = double(E) * log(double(E) / (double(E) * double(E)));
//    double result = bm - log_likelihood_p;
//    double mdl = entropy::null_mdl_v1(E);
//    EXPECT_FLOAT_EQ(mdl, result);
//    EXPECT_FLOAT_EQ(mdl, hand_calculated_mdl);
//}

//TEST_F(NonparametricEntropyTest, NullModelMDLv2ShouldGiveCorrectMDLForLargeGraph) {
//    double hand_calculated_mdl = 3.089407E+14;
//    long E = 5E12;
//    long V = 1E12;
//    auto blocks = double(V);
//    double x = (blocks * blocks) / double(E);
//    double h = ((1 + x) * log(1 + x)) - (x * log(x));
//    double bm = (double(E) * h) + double(V) * log(blocks);
//    double cell_value = double(E) / (blocks * blocks);
//    double cell_degree = double(E) / blocks;
//    double log_likelihood_p = (blocks * blocks) * (cell_value * log(cell_value / (cell_degree * cell_degree)));
//    double result = bm - log_likelihood_p;
//    double mdl = entropy::null_mdl_v2(V, E);
//    EXPECT_FLOAT_EQ(mdl, result);
//    EXPECT_FLOAT_EQ(mdl, hand_calculated_mdl);
//}

//TEST_F(NonparametricBlockMergeEntropyTest, BlockmodelDeltaMDLIsCorrectlyComputeWithBlockmodelDeltas) {
//    double E_before = entropy::nonparametric::mdl(B, graph);  // 11, 23);
//    double dE = entropy::nonparametric::block_merge_delta_mdl(B, { }, graph, Deltas);  // 0, B, Deltas, new_block_degrees);
//    double E_after = entropy::nonparametric::mdl(B2, graph);  // 11, 23);
//    EXPECT_FLOAT_EQ(E_after - E_before, dE);
//}

TEST_F(NonparametricBlockMergeEntropyTest, BlockmodelDeltaMDLIsCorrectlyComputeWithBlockmodelDeltasSansBlockDegrees) {
    double E_before = entropy::nonparametric::mdl(B, graph);  // 11, 23);
    utils::print<long>(B2.block_assignment());
    double dE = entropy::nonparametric::block_merge_delta_mdl(B, {1, B.degrees_out(0),
                                                       B.degrees_in(0), B.degrees(0)}, graph, Deltas);
    double E_after = entropy::nonparametric::mdl(B2, graph);  // 11, 23);
    EXPECT_FLOAT_EQ(E_after - E_before, dE);
}
