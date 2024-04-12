/**
 * Functions relating to calculating blockmodel entropy.
 */

#include "blockmodel/blockmodel.hpp"
#include "distributed/two_hop_blockmodel.hpp"
#include "common.hpp"
#include "delta.hpp"
#include "fastlgamma.hpp"
#include "utils.hpp"

namespace entropy {

const double BETA_DL = 1.0;

/// Computes the change in entropy under a proposed block merge.
[[deprecated("use blockmodel deltas without block degrees instead")]]
double block_merge_delta_mdl(long current_block, long proposal, long num_edges, const Blockmodel &blockmodel,
                             EdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees);

/// Computes the change in entropy under a proposed block merge using sparse intermediate structures.
[[deprecated("use blockmodel deltas without block degrees instead")]]
double block_merge_delta_mdl(long current_block, long proposal, long num_edges, const Blockmodel &blockmodel,
                             SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees);

/// Computes the change in entropy under a proposed block merge using changes to the blockmodel.
[[deprecated("use blockmodel deltas without block degrees instead")]]
double block_merge_delta_mdl(long current_block, const Blockmodel &blockmodel, const Delta &delta,
                             common::NewBlockDegrees &block_degrees);

/// Computes the change in entropy under a proposed block merge using changes to the blockmodel. This method should
/// be preferred in almost all cases.
double block_merge_delta_mdl(long current_block, utils::ProposalAndEdgeCounts proposal, const Blockmodel &blockmodel,
                             const Delta &delta);

/// Computes the change in blockmodel minimum description length when a vertex moves from `current_block` to `proposal`.
/// Uses a dense version of `updates` to the blockmodel, and requires pre-calculated updated `block_degrees`.
[[deprecated("use blockmodel deltas instead")]]
double delta_mdl(long current_block, long proposal, const Blockmodel &blockmodel, long num_edges,
                 EdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees);

/// Computes the change in blockmodel minimum description length when a vertex moves from `current_block` to `proposal`.
/// Uses a sparse version of `updates` to the blockmodel, and requires pre-calculated updated `block_degrees`.
[[deprecated("use blockmodel deltas instead")]]
double delta_mdl(long current_block, long proposal, const Blockmodel &blockmodel, long num_edges,
                 SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees);

/// Computes the change in blockmodel minimum description length when a vertex moves from one block to another. Uses
/// changes to the blockmodel, stored in `delta`, to perform the computation, and does not require pre-calculated
/// updated block_degrees. This method should be preferred in almost all cases.
double delta_mdl(const Blockmodel &blockmodel, const Delta &delta, const utils::ProposalAndEdgeCounts &proposal);

/// Computes the Hastings correction using dense vectors.
[[deprecated("use blockmodel deltas instead")]]
double hastings_correction(const Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           utils::ProposalAndEdgeCounts &proposal, EdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees);

/// Computes the Hastings correction using sparse vectors.
[[deprecated("use blockmodel deltas instead")]]
double hastings_correction(const Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           utils::ProposalAndEdgeCounts &proposal, SparseEdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees);

/// Computes the hastings correction using the blockmodel deltas under the proposed vertex move.
double hastings_correction(long vertex, const Graph &graph, const Blockmodel &blockmodel, const Delta &delta,
                           long current_block, const utils::ProposalAndEdgeCounts &proposal);

/// Calculates the minimum description length of `blockmodel` for a directed graph with `num_vertices` vertices and
/// `num_edges` edges.
double mdl(const Blockmodel &blockmodel, const Graph &graph);

/// Computes the normalized minimum description length using `null_mdl_v1`.
double normalize_mdl_v1(double mdl, const Graph &graph);

/// Computes the normalized minimum description length using `null_mdl_v2`.
[[deprecated("use normalize_mdl_v1 instead")]]
double normalize_mdl_v2(double mdl, long num_vertices, long num_edges);

/// Computes the minimum description length of the null model with only one block.
double null_mdl_v1(const Graph &graph);

/// Computes the minimum description length of the null model with as many blocks as there are vertices.
[[deprecated("use null_mdl_v1 instead")]]
double null_mdl_v2(long num_vertices, long num_edges);

// TODO: add an undirected mdl
// TODO: add undirected delta_mdl functions

namespace dist {

/// Computes the overall entropy of the given blockmodel for a directed graph.
double mdl(const TwoHopBlockmodel &blockmodel, long num_vertices, long num_edges);

// TODO: add an undirected distributed mdl

// TODO: handle case when number of edges/vertices >= 2.15 billion (long --> long)

}

namespace nonparametric {

inline double eterm_exact(long source, long destination, long weight) {
    double val = fastlgamma(weight + 1);

    if (args.undirected && source == destination) {
        double log_2 = log(2);
        return -val - weight * log_2;
    } else {
        return -val;
    }
}

inline double vterm_exact(long out_degree, long in_degree, long weight) { // out_degree, in_degree, wr=size of community, true? meh?
//    if (deg_corr)
//    {
//    if constexpr (is_directed_::apply<Graph>::type::value)
//        return fastlgamma(out_degree + 1) + fastlgamma(in_degree + 1);

    if (args.degreecorrected) {
        if (args.undirected)
            return fastlgamma(out_degree + 1);
        return fastlgamma(out_degree + 1) + fastlgamma(in_degree + 1);
    }

    if (weight == 0) return 0.0;
    if (args.undirected)
        return out_degree * fastlog(weight);
    return (out_degree + in_degree) * fastlog(weight);
//    }
//    else
//    {
//        if constexpr (is_directed_::apply<Graph>::type::value)
//            return (out_degree + in_degree) * safelog_fast(wr);
//        else
//            return out_degree * safelog_fast(wr);
//    }
}

double get_deg_entropy(const Graph &graph, long vertex);

double sparse_entropy(const Blockmodel &blockmodel, const Graph &graph);

inline double fastlbinom(long N, long k) {
    if (N == 0 || k == 0 || k > N)
        return 0;
    return ((fastlgamma(N + 1) - fastlgamma(k + 1)) - fastlgamma(N - k + 1));
}

double get_partition_dl(long N, const Blockmodel &blockmodel);

/// No idea what this function does. See int_part.cc in https://git.skewed.de/count0/graph-tool
double get_v(double u, double epsilon=1e-8);

double log_q_approx_small(size_t n, size_t k);

/// Computes the number of restricted of integer n into at most m parts. This is part of teh prior for the
/// degree-corrected SBM.
/// TO-DO: the current function contains only the approximation of log_q. If it becomes a bottleneck, you'll want to
/// compute a cache of log_q(n, m) for ~20k n and maybe a few hundred m? I feel like for larger graphs, the cache
/// will be a waste of time.
/// See int_part.cc in https://git.skewed.de/count0/graph-tool
double log_q(size_t n, size_t k);

double get_deg_dl_dist(const Blockmodel &blockmodel);

double get_edges_dl(size_t B, size_t E);

/// Computes the nonparametric description length of `blockmodel` given `graph`.
double mdl(const Blockmodel &blockmodel, const Graph &graph);

/// Obtain the entropy difference given a set of entries in the blockmodel matrix
double entries_dS(const Blockmodel &blockmodel, const Delta &delta);

/// Compute the entropy difference of a virtual move of vertex from one block to another
double virtual_move_sparse(const Blockmodel &blockmodel, const Delta &delta,
                           const utils::ProposalAndEdgeCounts &proposal);

double get_delta_partition_dl(long num_vertices, const Blockmodel &blockmodel, const Delta &delta, long weight);

double get_delta_deg_dl_dist_change(const Blockmodel &blockmodel, long block, long vkin, long vkout, long vweight,
                                    int diff);

double get_delta_deg_dl(long vertex, const Blockmodel &blockmodel, const Delta &delta, const Graph &graph);

double get_delta_edges_dl(const Blockmodel &blockmodel, const Delta &delta, long weight, long num_edges);

double delta_mdl(const Blockmodel &blockmodel, const Graph &graph, long vertex, const Delta &delta,
                 const utils::ProposalAndEdgeCounts &proposal);

double block_merge_delta_mdl(const Blockmodel &blockmodel, const utils::ProposalAndEdgeCounts &proposal,
                             const Graph &graph, const Delta &delta);

}

}