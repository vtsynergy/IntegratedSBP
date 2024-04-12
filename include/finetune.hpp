/**
 * The finetuning phase of the stochastic block blockmodeling algorithm.
 */
#ifndef SBP_FINETUNE_HPP
#define SBP_FINETUNE_HPP

#include <cmath>
#include <vector>

#include <omp.h>

#include "common.hpp"
#include "graph.hpp"
#include "blockmodel/blockmodel.hpp"
#include "blockmodel/blockmodel_triplet.hpp"
#include "blockmodel/sparse/delta.hpp"
#include "utils.hpp"
#include "typedefs.hpp"

/*******************
 * FINE-TUNE
 ******************/
namespace finetune {

/// The total number of MCMC iterations completed, to be dynamically updated during execution.
extern long MCMC_iterations;

/// The total amount of time spent performing MCMC iterations, to be dynamically updated during execution.
extern double MCMC_time;

/// The total amount of time spent in the main parallelizable loop of the MCMC iterations, to by dynamically
/// updated during execution.
extern double MCMC_sequential_time, MCMC_parallel_time, MCMC_vertex_move_time;

/// The number of MCMC moves performed.
extern ulong MCMC_moves;

//struct Neighbors {
//    EdgeWeights out_neighbors;
//    EdgeWeights in_neighbors;
//};

//static const long MOVING_AVG_WINDOW = 3;      // Window for calculating change in entropy
//static const double SEARCH_THRESHOLD = 5e-4; // Threshold before golden ratio is established
//static const double GOLDEN_THRESHOLD = 1e-4; // Threshold after golden ratio is established
static const long MAX_NUM_ITERATIONS = 100;   // Maximum number of finetuning iterations

bool accept(double delta_entropy, double hastings_correction);

Blockmodel &asynchronous_gibbs(Blockmodel &blockmodel, const Graph &graph, BlockmodelTriplet &blockmodels);

//Blockmodel &asynchronous_gibbs_v2(Blockmodel &blockmodel, const Graph &graph, BlockmodelTriplet &blockmodels);

EdgeWeights block_edge_weights(const std::vector<long> &block_assignment, const EdgeWeights &neighbor_weights);

/// Returns the potential changes to the blockmodel if the vertex with `out_edges` and `in_edges` moves from
/// `current_block` into `proposed_block`.
/// NOTE: assumes that any self edges are present in exactly one of `out_edges` and `in_edges`.
Delta blockmodel_delta(long vertex, long current_block, long proposed_block, const EdgeWeights &out_edges,
                       const EdgeWeights &in_edges, const Blockmodel &blockmodel);

/// Counts the number of neighboring blocks for low degree vertices in the graph. Used for load balancing.
std::pair<std::vector<long>, long> count_low_degree_block_neighbors(const Graph &graph, const Blockmodel &blockmodel);

bool early_stop(long iteration, BlockmodelTriplet &blockmodels, double initial_entropy,
                std::vector<double> &delta_entropies);

bool early_stop(long iteration, double initial_entropy, std::vector<double> &delta_entropies);

bool early_stop_parallel(long iteration, BlockmodelTriplet &blockmodels, double initial_entropy,
                         std::vector<double> &delta_entropies, std::vector<long> &vertex_moves);

[[maybe_unused]] EdgeCountUpdates edge_count_updates(ISparseMatrix *blockmodel, long current_block, long proposed_block,
                                                     EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                                     long self_edge_weight);

void edge_count_updates_sparse(const Blockmodel &blockmodel, long vertex, long current_block, long proposed_block,
                               EdgeWeights &out_blocks, EdgeWeights &in_blocks, SparseEdgeCountUpdates &updates);

/// Returns the edge weights in `neighbors[vertex]` as an `EdgeWeights` struct. If `ignore_self` is `true`, then
/// self-edges will not be added to EdgeWeights.
EdgeWeights edge_weights(const NeighborList &neighbors, long vertex, bool ignore_self = false);

/// Evaluates a potential move of `vertex` from `current_block` to `proposal.proposal` using MCMC logic.
VertexMove eval_vertex_move(long vertex, long current_block, utils::ProposalAndEdgeCounts proposal,
                            const Blockmodel &blockmodel, const Graph &graph, EdgeWeights &out_edges,
                            EdgeWeights &in_edges);

/// Evaluates a potential move of `vertex` from `current_block` to `proposal.proposal` using MCMC logic.
VertexMove_v2 eval_vertex_move_v2(long vertex, long current_block, utils::ProposalAndEdgeCounts proposal,
                                  const Blockmodel &blockmodel, const Graph &graph, EdgeWeights &out_edges,
                                  EdgeWeights &in_edges);

/// Evaluates a potential move of `vertex` from `current_block` to `proposal.proposal` using MCMC logic without using
/// blockmodel deltas.
//VertexMove eval_vertex_move_nodelta(long vertex, long current_block, utils::ProposalAndEdgeCounts proposal,
//                                    const Blockmodel &blockmodel, const Graph &graph, EdgeWeights &out_edges,
//                                    EdgeWeights &in_edges);

/// Runs the synchronous Metropolis Hastings algorithm on the high-degree vertices of `blockmodel`, and
/// Asynchronous Gibbs on the rest.
Blockmodel &hybrid_mcmc(Blockmodel &blockmodel, const Graph &graph, BlockmodelTriplet &blockmodels);

/// Runs the synchronous Metropolis Hastings algorithm on the high-degree vertices of `blockmodel`, and
/// Asynchronous Gibbs on the rest. Attempts to manually balance the workload using number of block neighbors.
Blockmodel &hybrid_mcmc_load_balanced(Blockmodel &blockmodel, const Graph &graph, BlockmodelTriplet &blockmodels);

[[maybe_unused]] Blockmodel &finetune_assignment(Blockmodel &blockmodel, Graph &graph);

/// Returns a vector which determines which blocks a thread is responsible for.
std::vector<bool> load_balance(const Blockmodel &blockmodel, const std::vector<std::pair<long, long>> &block_properties);

/// Returns a vector which determines which vertices a thread is responsible for.
std::vector<bool> load_balance_vertices(const Graph &graph, const std::vector<std::pair<long, long>> &vertex_properties);

/// Returns a vector which determines which vertices a thread is responsible for, using block neighbors.
std::vector<bool> load_balance_block_neighbors(const Graph &graph, const Blockmodel &blockmodel,
                                               const std::pair<std::vector<long>, long> &block_neighbors);

/// Runs the synchronous Metropolis Hastings algorithm on `blockmodel`.
Blockmodel &metropolis_hastings(Blockmodel &blockmodel, const Graph &graph, BlockmodelTriplet &blockmodels);

/// Moves `vertex` from `current_block` to `proposal.proposal` using MCMC logic.
VertexMove move_vertex(long vertex, long current_block, utils::ProposalAndEdgeCounts proposal, Blockmodel &blockmodel,
                       const Graph &graph, EdgeWeights &out_edges, EdgeWeights &in_edges);

/// Moves `vertex` from `current_block` to `proposal.proposal` using MCMC logic without using blockmodel deltas.
//VertexMove move_vertex_nodelta(long vertex, long current_block, utils::ProposalAndEdgeCounts proposal,
//                               Blockmodel &blockmodel, const Graph &graph, EdgeWeights &out_edges,
//                               EdgeWeights &in_edges);

/// Computes the overall entropy of the given blockmodel.
//double mdl(const Blockmodel &blockmodel, long num_vertices, long num_edges);

/// Proposes a new Metropolis-Hastings vertex move.
VertexMove propose_move(Blockmodel &blockmodel, long vertex, const Graph &graph);

/// Proposes a new Asynchronous Gibbs vertex move.
VertexMove propose_gibbs_move(const Blockmodel &blockmodel, long vertex, const Graph &graph);

/// Proposes a new Asynchronous Gibbs vertex move.
VertexMove_v2 propose_gibbs_move_v2(const Blockmodel &blockmodel, long vertex, const Graph &graph);

/// Proposes a new Asynchronous Gibbs vertex move. Contains additional information needed for nonparametric entropy
/// computations. Should be preferred over _v2.
VertexMove_v3 propose_gibbs_move_v3(const Blockmodel &blockmodel, long vertex, const Graph &graph);

/// Sorts blocks in order of number of neighbors - used for load balancing.
std::vector<std::pair<long,long>> sort_blocks_by_neighbors(const Blockmodel &blockmodel);

/// Sorts blocks in order of block size - used for load balancing.
std::vector<std::pair<long,long>> sort_blocks_by_size(const Blockmodel &blockmodel);

/// Sorts vertices in order of degree - used for load balancing.
std::vector<std::pair<long,long>> sort_vertices_by_degree(const Graph &graph);

//namespace directed {
//
///// Computes the overall entropy of the given blockmodel for a directed graph.
//double overall_entropy(const Blockmodel &blockmodel, long num_vertices, long num_edges);
//
//}  // namespace directed
//
//namespace undirected {
//
///// Computes the overall entropy of the given blockmodel for an undirected graph.
//double overall_entropy(const Blockmodel &blockmodel, long num_vertices, long num_edges);
//
//}  // namespace undirected

} // namespace finetune

#endif // SBP_FINETUNE_HPP
