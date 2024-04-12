/***
 * Stores the current graph blockmodeling results.
 */
#ifndef SBP_DELTA_HPP
#define SBP_DELTA_HPP

#include <iostream>
#include <limits>
#include <numeric>
#include <map>
#include <memory>
#include <queue>

// #include <Eigen/Core>
// #include "sparse/boost_mapped_matrix.hpp"
#include "../args.hpp"
#include "delta.hpp"
#include "sparse/dict_matrix.hpp"
#include "sparse/dict_transpose_matrix.hpp"
#include "../graph.hpp"
#include "sparse/typedefs.hpp"
#include "../utils.hpp"
#include "typedefs.hpp"
#include "utils.hpp"

static const double BLOCK_REDUCTION_RATE = 0.5;

// The time taken to build the blockmodel
extern double BLOCKMODEL_BUILD_TIME;
extern double Blockmodel_sort_time;
extern double Blockmodel_access_time;
extern double Blockmodel_update_assignment;

// typedef py::EigenDRef<Eigen::Matrix<long, Eigen::Dynamic, 2>> Matrix2Column;

typedef struct edge_count_updates_t {
    std::vector<long> block_row;
    std::vector<long> proposal_row;
    std::vector<long> block_col;
    std::vector<long> proposal_col;
} EdgeCountUpdates;

typedef struct sparse_edge_count_updates_t {
    MapVector<long> block_row;
    MapVector<long> proposal_row;
    MapVector<long> block_col;
    MapVector<long> proposal_col;
} SparseEdgeCountUpdates;

// TODO: make a Blockmodel interface (?) Or keep Blockmodel pointers in memory
class Blockmodel {
  public:
    Blockmodel() : empty(true) {
        this->num_blocks = 0;
        this->block_reduction_rate = 0.0;
        this->num_blocks_to_merge = 0;
        this->_num_nonempty_blocks = 0;
        this->overall_entropy = std::numeric_limits<double>::max();
    }
    Blockmodel(long num_blocks, double block_reduction_rate) : empty(false) {
        this->num_blocks = num_blocks;
        this->block_reduction_rate = block_reduction_rate;
        this->overall_entropy = std::numeric_limits<double>::max();
        this->_num_nonempty_blocks = num_blocks;
        if (args.transpose) {
            this->_blockmatrix = std::make_shared<DictTransposeMatrix>(this->num_blocks, this->num_blocks, 36);
        } else {
            this->_blockmatrix = std::make_shared<DictMatrix>(this->num_blocks, this->num_blocks);
        }
        // Set the block assignment to be the range [0, this->num_blocks)
        this->_block_assignment = utils::range<long>(0, this->num_blocks);
        // Set the block sizes to be 0 (empty blocks)
        this->_block_sizes = utils::constant<long>(this->num_blocks, 0);
        // Number of blocks to merge
        this->num_blocks_to_merge = (long)(this->num_blocks * this->block_reduction_rate);
    }
    Blockmodel(long num_blocks, const Graph &graph, double block_reduction_rate)
        : Blockmodel(num_blocks, block_reduction_rate) {
        this->initialize_edge_counts(graph);
    }
    Blockmodel(long num_blocks, const Graph &graph, double block_reduction_rate,
               std::vector<long> &block_assignment) : Blockmodel(num_blocks, block_reduction_rate) {
        // Set the block assignment
        this->_block_assignment = block_assignment;
        // Number of blocks to merge
        this->initialize_edge_counts(graph);
    }
    /// Returns an immutable copy of the vertex-to-block assignment vector.
    const std::vector<long> &block_assignment() const { return this->_block_assignment; }
    /// Returns the block assignment for `vertex`.
    long block_assignment(long vertex) const { return this->_block_assignment[vertex]; }
    /// Returns the vector of block sizes.
    const std::vector<long>& block_sizes() const { return this->_block_sizes; }
    /// Returns the number of vertices in block `block`.
    long block_size(long block) const { return this->_block_sizes[block]; }
    /// Returns the normalized difference in block sizes.
    double block_size_variation() const;
    /// TODO
    static std::vector<long> build_mapping(const std::vector<long> &values) ;
    /// Performs the block merges with the highest change in entropy/MDL
    void carry_out_best_merges(const std::vector<double> &delta_entropy_for_each_block,
                               const std::vector<long> &best_merge_for_each_block);
    /// TODO
    Blockmodel clone_with_true_block_membership(const Graph &graph, std::vector<long> &true_block_membership);
    /// Returns a copy of the current Blockmodel
    Blockmodel copy();
    /// TODO documentation
    // TODO: move block_reduction_rate to some constants file
    static Blockmodel from_sample(long num_blocks, const Graph &graph, std::vector<long> &sample_block_membership,
                                 std::map<long, long> &mapping, double block_reduction_rate);
    /// Difficulty score, being the geometric mean between block_size_variation() and interblock_edges().
    double difficulty_score() const;
    /// Fills the blockmodel using the edges in `graph` and the current vertex-to-block `block_assignment`.
    void initialize_edge_counts(const Graph &graph);
    /// TODO
    double log_posterior_probability() const;
    /// TODO
    double log_posterior_probability(long num_edges) const;
    /// Merges block `merge_from` into block `merge_to`
    void merge_block(long merge_from, long merge_to, const Delta &delta, utils::ProposalAndEdgeCounts proposal);
    /// Moves `vertex` from `current_block` to `new_block`. Updates the blockmodel using the new rows and columns from
    /// `updates`, and updates the block degrees.
    /// TODO: update block degrees on the fly.
    void move_vertex(Vertex vertex, long current_block, long new_block, EdgeCountUpdates &updates,
                     std::vector<long> &new_block_degrees_out, std::vector<long> &new_block_degrees_in,
                     std::vector<long> &new_block_degrees);
    /// Moves `vertex` from `current_block` to `new_block`. Updates the blockmodel using the new rows and columns from
    /// `updates`, and updates the block degrees.
    /// TODO: update block degrees on the fly.
    void move_vertex(Vertex vertex, long current_block, long new_block, SparseEdgeCountUpdates &updates,
                     std::vector<long> &new_block_degrees_out, std::vector<long> &new_block_degrees_in,
                     std::vector<long> &new_block_degrees);
    /// Moves `vertex` from `current_block` to `new_block`. Updates the blockmodel using the new blockmodel values from
    /// `delta`, and updates the block degrees.
    /// TODO: update block degrees on the fly.
    void move_vertex(Vertex vertex, long new_block, const Delta &delta, std::vector<long> &new_block_degrees_out,
                     std::vector<long> &new_block_degrees_in, std::vector<long> &new_block_degrees);
    /// Moves `vertex` from one block to another. Updates the blockmodel using the new blockmodel values from `delta`,
    /// and updates the block degrees, which are calculated on-the-fly.
    void move_vertex(Vertex vertex, const Delta &delta, utils::ProposalAndEdgeCounts &proposal);
    /// Moves a vertex from one block to another. Updates the blockmodel based on the edges in `move`,
    /// and updates the block degrees, which are calculated on-the-fly. NOTE: assumes self-edges are only included in
    /// move.out_edges.
    void move_vertex(const VertexMove_v3 &move);
    /// TODO
    void set_block_membership(long vertex, long block);
    /// TODO: Get rid of getters and setters?
    std::shared_ptr<ISparseMatrix> blockmatrix() const { return this->_blockmatrix; }
//    ISparseMatrix *blockmatrix() const { return this->_blockmatrix; }
    /// Returns true if `block1` is a neighbor of `block2`.
    bool is_neighbor_of(long block1, long block2) const;
    /// Returns the percentage of edges occurring between blocks.
    double interblock_edges() const;
    /// prints blockmatrix to file (should not be used for large blockmatrices)
    void print_blockmatrix() const;
    /// prints the blockmodel with some additional information.
    void print_blockmodel() const;
    /// Returns true if the blockmodel owns the current block (always returns true for non-distributed blockmodel).
    bool stores(long block) const { return true; }
    /// TODO
    void update_block_assignment(long from_block, long to_block);
    /// Updates the blockmodel values for `current_block` and `proposed_block` using the rows and columns in `updates`.
    void update_edge_counts(long current_block, long proposed_block, EdgeCountUpdates &updates);
    /// Updates the blockmodel values for `current_block` and `proposed_block` using the rows and columns in `updates`.
    void update_edge_counts(long current_block, long proposed_block, SparseEdgeCountUpdates &updates);
    /// Validates the blockmatrix entries given the current block assignment.
    bool validate(const Graph &graph) const;
    /// Sets the block assignment for this `vertex` to `block`.
    void set_block_assignment(long vertex, long block) { this->_block_assignment[vertex] = block; }
    void set_block_assignment(std::vector<long> block_assignment) { this->_block_assignment = block_assignment; }
    const std::vector<long> &degrees() const { return this->_block_degrees; }
    long degrees(long block) const { return this->_block_degrees[block]; }
    void degrees(long block, long value) { this->_block_degrees[block] = value; }
    void degrees(std::vector<long> block_degrees) { this->_block_degrees = block_degrees; }
    const std::vector<long> &degrees_in() const { return this->_block_degrees_in; }
    long degrees_in(long block) const { return this->_block_degrees_in[block]; }
    void degrees_in(long block, long value) { this->_block_degrees_in[block] = value; }
    void degrees_in(std::vector<long> block_degrees_in) { this->_block_degrees_in = block_degrees_in; }
    const std::vector<long> &degrees_out() const { return this->_block_degrees_out; }
    long degrees_out(long block) const { return this->_block_degrees_out[block]; }
    void degrees_out(long block, long value) { this->_block_degrees_out[block] = value; }
    void degrees_out(std::vector<long> block_degrees_out) { this->_block_degrees_out = block_degrees_out; }
    long num_nonempty_blocks() const { return this->_num_nonempty_blocks; }
    /// Returns the out-degree histogram for block `block`.
    const MapVector<long> &out_degree_histogram(long block) const { return this->_out_degree_histogram[block]; }
    /// Returns the in-degree histogram for block `block`.
    const MapVector<long> &in_degree_histogram(long block) const { return this->_in_degree_histogram[block]; }
    double &getBlock_reduction_rate() { return this->block_reduction_rate; }
    void setBlock_reduction_rate(double block_reduction_rate) { this->block_reduction_rate = block_reduction_rate; }
    double getOverall_entropy() const { return this->overall_entropy; }
    void setOverall_entropy(double overall_entropy) { this->overall_entropy = overall_entropy; }
    long &getNum_blocks_to_merge() { return this->num_blocks_to_merge; }
    void setNum_blocks_to_merge(long num_blocks_to_merge) { this->num_blocks_to_merge = num_blocks_to_merge; }
    long getNum_blocks() const { return this->num_blocks; }
    void setNum_blocks(long num_blocks) { this->num_blocks = num_blocks; }
    // Other
    bool empty;

  protected:
    // Structure
    long num_blocks;
    long _num_nonempty_blocks;
    std::shared_ptr<ISparseMatrix> _blockmatrix;
//    ISparseMatrix *_blockmatrix;
    // Known info
    std::vector<long> _block_assignment;
    std::vector<long> _block_degrees;
    std::vector<long> _block_degrees_in;
    std::vector<long> _block_degrees_out;
    std::vector<long> _block_sizes;
    std::vector<MapVector<long>> _out_degree_histogram;
    std::vector<MapVector<long>> _in_degree_histogram;
    double block_reduction_rate;
    // Computed info
    double overall_entropy;
    long num_blocks_to_merge;
};

#endif // SBP_DELTA_HPP
