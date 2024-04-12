/**
 * The distributed stochastic block blockmodeling module.
 */
#ifndef SBP_DIST_SBP_HPP
#define SBP_DIST_SBP_HPP

#include "args.hpp"
#include "blockmodel.hpp"
#include "distributed/dist_blockmodel_triplet.hpp"
#include "distributed/two_hop_blockmodel.hpp"
#include "graph.hpp"
#include "sbp.hpp"

namespace sbp::dist {

// The amount of time taken to finetune the partition.
extern double finetune_time;

/// Adds intermediate results to be later saved in a CSV file.
void add_intermediate(double iteration, const Graph &graph, double modularity, double mdl);

std::vector<intermediate> get_intermediates();

/// Performs community detection on the provided graph using MPI, using the stochastic block partitioning algorithm
Blockmodel stochastic_block_partition(Graph &graph, Args &args, bool divide_and_conquer = false);

/// Returns true if the exit condition is reached based on the provided distributed blockmodels
bool done_blockmodeling(TwoHopBlockmodel &blockmodel, DistBlockmodelTriplet &blockmodel_triplet,
                        long min_num_blocks = 0);

} // namespace sbp::dist

#endif  // SBP_DIST_SBP_HPP