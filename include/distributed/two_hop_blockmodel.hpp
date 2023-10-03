/// ====================================================================================================================
/// Part of the accelerated Stochastic Block Partitioning (SBP) project.
/// Copyright (C) Virginia Polytechnic Institute and State University, 2023. All Rights Reserved.
///
/// This software is provided as-is. Neither the authors, Virginia Tech nor Virginia Tech Intellectual Properties, Inc.
/// assert, warrant, or guarantee that the software is fit for any purpose whatsoever, nor do they collectively or
/// individually accept any responsibility or liability for any action or activity that results from the use of this
/// software.  The entire risk as to the quality and performance of the software rests with the user, and no remedies
/// shall be provided by the authors, Virginia Tech or Virginia Tech Intellectual Properties, Inc.
/// This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
/// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
/// details.
/// You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to
/// the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.
///
/// Author: Frank Wanye
/// ====================================================================================================================
/***
 * Stores the current distributed graph blockmodeling results.
 */
#ifndef SBP_DIST_BLOCKMODEL_HPP
#define SBP_DIST_BLOCKMODEL_HPP

#include <iostream>
#include <limits>
#include <numeric>
#include <map>
#include <memory>
#include <mpi.h>
#include <random>
#include <queue>

#include "../args.hpp"
#include "blockmodel.hpp"
#include "../graph.hpp"
#include "mpi_data.hpp"
#include "sparse/dict_matrix.hpp"
#include "sparse/dict_transpose_matrix.hpp"
#include "sparse/typedefs.hpp"
#include "../utils.hpp"

extern double Load_balancing_time;

extern std::vector<long> Rank_indices;

class TwoHopBlockmodel : public Blockmodel {
public:
    // using Blockmodel::Blockmodel;
    // Constructors are not derived from base class
    TwoHopBlockmodel() : Blockmodel() {}
    TwoHopBlockmodel(long num_blocks, double block_reduction_rate) : Blockmodel(num_blocks, block_reduction_rate) {}
    TwoHopBlockmodel(long num_blocks, const Graph &graph, double block_reduction_rate)
            : TwoHopBlockmodel(num_blocks, block_reduction_rate) {
        // If the block assignment is not provided, use round-robin assignment
        this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
        for (long i = mpi.rank; i < this->num_blocks; i += mpi.num_processes) {  // round-robin work mapping
            this->_my_blocks[i] = true;
        }
        this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, true);  // no distribution
        this->initialize_edge_counts(graph);
    }
    TwoHopBlockmodel(long num_blocks, const Graph &graph, double block_reduction_rate,
                     std::vector<long> &block_assignment) : TwoHopBlockmodel(num_blocks, block_reduction_rate) {
        // Set the block assignment
        this->_block_assignment = block_assignment;
        this->distribute(graph);
        this->initialize_edge_counts(graph);
    }
    /// Sets the _in_two_hop_radius for a 2-hop blockmodel.
    void build_two_hop_blockmodel(const NeighborList &neighbors);
    TwoHopBlockmodel copy();
    /// Distributes the blockmodel amongst MPI ranks. Needs to be called before the first call to
    /// initialize_edge_counts, since it sets the _in_two_hop_radius and _my_blocks vectors. After that, it only needs
    /// to be called to re-distribute the blockmodel (followed by initialize_edge_counts).
    void distribute(const Graph &graph);
    /// Returns the _in_two_hop_radius vector.
    const std::vector<bool>& in_two_hop_radius() const { return this->_in_two_hop_radius; }
    void initialize_edge_counts(const Graph &graph);
    double log_posterior_probability() const;
    /// Returns true if this blockmodel owns the compute for the requested block.
    bool owns_block(long block) const;
    /// Returns true if this blockmodel owns the compute for the requested vertex.
    bool owns_vertex(long vertex) const;
    /// Returns true if this blockmodel owns storage for the requested block.
    bool stores(long block) const;
    bool validate(const Graph &graph) const;
private:
    // ===== Functions
    /// Returns a sorted vector of <block, block size> pairs, in descending order of block size.
    std::vector<std::pair<long,long>> sorted_block_sizes() const;
    /// No data distribution, work on blocks is mapped using round-robin strategy.
    void distribute_none();
    /// No data distribution, work on vertices is mapped to try to distribute aggregate block degree amongst MPI ranks.
    /// That is, it tries to distribute the vertices based on the block degree of the blocks they belong to. Vertices
    /// and blocks are distributed separately, which is fine because the entire blockmodel is replicated on each rank.
    void distribute_none_agg_block_degree_balanced(const Graph &graph);
    /// No data distribution, work on blocks is mapped to try to distribute an equal number of blocks amongst
    /// MPI ranks. Vertices and blocks are distributed together
    void distribute_none_block_degree_balanced(const Graph &graph);
    /// No data distribution, work on vertices is mapped to try to distribute an equal number of edges amongst
    /// MPI ranks. Vertices and blocks are distributed separately, which is fine because the entire blockmodel is
    /// replicated on each rank.
    void distribute_none_edge_balanced(const Graph &graph);
    // ===== Variables
    /// Stores true for in_two_hop_radius[block] if block is stored in this blockmodel.
    std::vector<bool> _in_two_hop_radius;
    /// Stores true for my_blocks[block] if this blockmodel owns the compute for this block.
    std::vector<bool> _my_blocks;
    /// Stores 1 for any vertex that this blockmodel owns.
    std::vector<long> _my_vertices;
};

#endif // SBP_DIST_BLOCKMODEL_HPP