/**
 * Structs and functions common to both the block merge and finetune phases.
 */

#ifndef SBP_COMMON_HPP
#define SBP_COMMON_HPP

#include <random>
#include <vector>

// #include <Eigen/Core>

#include "blockmodel/blockmodel.hpp"
#include "blockmodel/sparse/csparse_matrix.hpp"
#include "blockmodel/sparse/typedefs.hpp"
#include "fastlog.hpp"
#include "rng.hpp"
#include "utils.hpp"
#include "typedefs.hpp"

// TODO: move everything that uses `blockmodel` to one of the Blockmodel classes

namespace common {

extern std::uniform_int_distribution<long> candidates;

// TODO: explore making these static thread_local variables? Or create an array of these, with one per thread
//static std::random_device seeder;
//static std::mt19937 generator(seeder());
//static std::uniform_real_distribution<double> rng_distribution(0.0, 1.0);

typedef struct new_block_degrees_t {
    std::vector<long> block_degrees_out;
    std::vector<long> block_degrees_in;
    std::vector<long> block_degrees;
} NewBlockDegrees;

/// Calculates the entropy of a single blockmodel cell.
inline double cell_entropy(size_t value, size_t degree_in, size_t degree_out) {
    if (value == 0) return 0.0;
    double entropy = double(value) * (fastlog(value) - (fastlog(degree_in) + fastlog(degree_out)));
//    double entropy = value * logf(value / (degree_in * degree_out));
    if (std::isnan(entropy) || std::isinf(entropy)) {
        std::cerr << "ERROR " << "value: " << value << " dIn: " << degree_in << " dOut: " << degree_out << std::endl;
        throw std::invalid_argument("something is wrong");
    }
    return entropy;
    // return value * std::log(value / degree_in / degree_out);
}

/// TODO
long choose_neighbor(std::vector<long> &neighbor_indices, std::vector<long> &neighbor_weights);

/// Chooses a neighboring block using a multinomial distribution based on the number of edges connecting the current
/// block to the neighboring blocks.
long choose_neighbor(const SparseVector<double> &multinomial_distribution);

/// TODO: computing current_block_self_edges is annoying af. Maybe use Updates and Deltas instead.
/// Computes the block degrees under a proposed move
NewBlockDegrees compute_new_block_degrees(long current_block, const Blockmodel &blockmodel, long current_block_self_edges,
                                          long proposed_block_self_edges, utils::ProposalAndEdgeCounts &proposal);

/// Computes the entropy of one row or column of data.
double delta_entropy_temp(std::vector<long> &row_or_col, std::vector<long> &block_degrees, long degree, long num_edges);

// /// Computes the entropy of one row or column of sparse data
// double delta_entropy_temp(const MapVector<long> &row_or_col, const std::vector<long> &_block_degrees, long degree);

/// Computes the entropy of one row or column of sparse data.
double delta_entropy_temp(const MapVector<long> &row_or_col, const std::vector<long> &block_degrees, long degree,
                          long num_edges);

// /// Computes the entropy of one row or column of sparse data, ignoring indices `current_block` and `proposal`
// double delta_entropy_temp(const MapVector<long> &row_or_col, const std::vector<long> &_block_degrees, long degree,
//                           long current_block, long proposal);

/// Computes the entropy of one row or column of sparse data, ignoring indices `current_block` and `proposal`.
double delta_entropy_temp(const MapVector<long> &row_or_col, const std::vector<long> &block_degrees, long degree,
                          long current_block, long proposal, long num_edges);

/// Removes entries from in whose index is index1 or index
std::vector<long> exclude_indices(const std::vector<long> &in, long index1, long index2);

/// Removes entries from in whose index is index1 or index
MapVector<long>& exclude_indices(MapVector<long> &in, long index1, long index2);

/// Returns a subset of <values> corresponding to the indices where the value of <indices> != 0
std::vector<long> index_nonzero(const std::vector<long> &values, std::vector<long> &indices);

/// Returns a subset of <values> corresponding to the indices where the value of <indices> != 0
std::vector<long> index_nonzero(const std::vector<long> &values, MapVector<long> &indices);

/// Returns the non-zero values in <in>
std::vector<long> nonzeros(std::vector<long> &in);

/// Returns the non-zero values in <in>
std::vector<long> nonzeros(MapVector<long> &in);

/// Proposes a new block for either the block merge or finetune step based on `bool block_merge`.
utils::ProposalAndEdgeCounts propose_new_block(long current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                               const std::vector<long> &block_assignment, const Blockmodel &blockmodel,
                                               bool block_merge = false);
/// TODO
long propose_random_block(long current_block, long num_blocks);

/// Returns a random integer between low and high
long random_integer(long low, long high);

namespace directed {

/// Computes the entropy of one row or column of data for a directed graph.
double delta_entropy_temp(std::vector<long> &row_or_col, std::vector<long> &block_degrees, long degree);

/// Computes the entropy of one row or column of sparse data for a directed graph.
double delta_entropy_temp(const MapVector<long> &row_or_col, const std::vector<long> &block_degrees, long degree);

/// Computes the entropy of one row or column of sparse data, ignoring indices `current_block` and `proposal`, for a
/// directed graph.
double delta_entropy_temp(const MapVector<long> &row_or_col, const std::vector<long> &block_degrees, long degree,
                          long current_block, long proposal);
}

namespace undirected {

/// Computes the entropy of one row or column of data for an undirected graph.
double delta_entropy_temp(std::vector<long> &row_or_col, std::vector<long> &block_degrees, long degree, long num_edges);

/// Computes the entropy of one row or column of sparse data for an undirected graph.
double delta_entropy_temp(const MapVector<long> &row_or_col, const std::vector<long> &block_degrees, long degree,
                          long num_edges);

/// Computes the entropy of one row or column of sparse data, ignoring indices `current_block` and `proposal`, for an
/// undirected graph.
double delta_entropy_temp(const MapVector<long> &row_or_col, const std::vector<long> &block_degrees, long degree,
                          long current_block, long proposal, long num_edges);
}

} // namespace common

#endif // SBP_COMMON_HPP
