#include "distributed/two_hop_blockmodel.hpp"

#include <unordered_set>

double Load_balancing_time = 0.0;

std::vector<long> Rank_indices;

void TwoHopBlockmodel::build_two_hop_blockmodel(const NeighborList &neighbors) {
    if (args.distribute == "none" || args.distribute == "none-edge-balanced" ||
        args.distribute == "none-agg-block-degree-balanced") {
        this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, true);
        return;
    }
    if (args.distribute == "2hop-snowball") {
        this->_my_blocks = std::vector<bool>(this->num_blocks, false);
        for (long v = 0; v < (long) neighbors.size(); ++v) {
            if (this->owns_vertex(v)) {
                long b = this->block_assignment(v);
                this->_my_blocks[b] = true;
            }
        }
    }
    // I think there will be a missing block in mcmc phase vertex->neighbor->block->neighbor_block
    this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, false);
    for (ulong vertex = 0; vertex < neighbors.size(); ++vertex) {
        std::vector<long> vertex_neighbors = neighbors[vertex];
        if (vertex_neighbors.empty()) {
            continue;
        }
        long block = this->_block_assignment[vertex];
        for (size_t i = 0; i < vertex_neighbors.size(); ++i) {
            long neighbor = vertex_neighbors[i];
            long neighbor_block = this->_block_assignment[neighbor];
            if (this->_my_blocks[block] || this->_my_blocks[neighbor_block]) {
            // if ((block % mpi.num_processes == mpi.rank) || (neighbor_block % mpi.num_processes == mpi.rank)) {
                this->_in_two_hop_radius[block] = true;
                this->_in_two_hop_radius[neighbor_block] = true;
            }
        }
    }
    long two_hop_radius_size = 0;
    for (const bool val : this->_in_two_hop_radius) {
        if (val) two_hop_radius_size++;
    }
    if (mpi.rank == 0) std::cout << "rank 0: num blocks in 2-hop radius == " << two_hop_radius_size << " / " << this->num_blocks << std::endl;
}

TwoHopBlockmodel TwoHopBlockmodel::copy() {
    TwoHopBlockmodel blockmodel_copy = TwoHopBlockmodel(this->num_blocks, this->block_reduction_rate);
    blockmodel_copy._block_assignment = std::vector<long>(this->_block_assignment);
    blockmodel_copy.overall_entropy = this->overall_entropy;
    blockmodel_copy._blockmatrix = std::shared_ptr<ISparseMatrix>(this->_blockmatrix->copy());
    blockmodel_copy._block_degrees = std::vector<long>(this->_block_degrees);
    blockmodel_copy._block_degrees_out = std::vector<long>(this->_block_degrees_out);
    blockmodel_copy._block_degrees_in = std::vector<long>(this->_block_degrees_in);
    blockmodel_copy._block_sizes = std::vector<long>(this->_block_sizes);
    blockmodel_copy._in_degree_histogram = std::vector<MapVector<long>>(this->_in_degree_histogram);
    blockmodel_copy._out_degree_histogram = std::vector<MapVector<long>>(this->_out_degree_histogram);
    blockmodel_copy._num_nonempty_blocks = this->_num_nonempty_blocks;
    blockmodel_copy._in_two_hop_radius = std::vector<bool>(this->_in_two_hop_radius);
    blockmodel_copy.num_blocks_to_merge = 0;
    blockmodel_copy._my_blocks = std::vector<bool>(this->_my_blocks);
    blockmodel_copy.empty = this->empty;
    return blockmodel_copy;
}

void TwoHopBlockmodel::distribute(const Graph &graph) {
    double start = MPI_Wtime();
    if (args.distribute == "none")
        distribute_none();
    else if (args.distribute == "2hop-round-robin")
        distribute_2hop_round_robin(graph.out_neighbors());
    else if (args.distribute == "2hop-size-balanced")
        distribute_2hop_size_balanced(graph.out_neighbors());
    else if (args.distribute == "2hop-snowball")
        distribute_2hop_snowball(graph.out_neighbors());
    else if (args.distribute == "none-edge-balanced")
        distribute_none_edge_balanced(graph);
    else if (args.distribute == "none-block-degree-balanced")
        distribute_none_block_degree_balanced(graph);
    else if (args.distribute == "none-agg-block-degree-balanced")
        distribute_none_agg_block_degree_balanced(graph);
    else
        distribute_none();
    if (args.distribute != "none" && args.distribute != "none-edge-balanced" &&
        args.distribute != "none-block-degree-balanced" &&
        args.distribute != "none-agg-block-degree-balanced") {
        std::cout << "WARNING: data distribution is NOT fully supported yet. "
                  << "We STRONGLY recommend running this software with --distribute none instead" << std::endl;
    }
    Load_balancing_time += MPI_Wtime() - start;
}

void TwoHopBlockmodel::distribute_none() {
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    for (long i = mpi.rank; i < this->num_blocks; i += mpi.num_processes)
        this->_my_blocks[i] = true;
    this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, true);
}

void TwoHopBlockmodel::distribute_none_edge_balanced(const Graph &graph) {
    if (Rank_indices.empty()) {
        if (mpi.rank == 0) std::cout << mpi.rank << " | rebuilding rank indices! =============" << std::endl;
        Rank_indices = utils::constant<long>(graph.num_vertices(), 0);
        std::vector<long> vertex_degrees = graph.degrees();
	    std::vector<long> sorted_indices = utils::argsort<long>(vertex_degrees);
        // std::vector<long> sorted_indices = utils::argsort(vertex_degrees);
        for (long i = mpi.rank; i < graph.num_vertices(); i += 2 * mpi.num_processes) {
            long vertex = sorted_indices[i];
            Rank_indices[vertex] = 1;
        }
        for (long i = 2 * mpi.num_processes - 1 - mpi.rank; i < graph.num_vertices(); i += 2 * mpi.num_processes) {
            long vertex = sorted_indices[i];
            Rank_indices[vertex] = 1;
        }
    }
    this->_my_vertices = Rank_indices;
    for (int rank = 0; rank < mpi.num_processes; ++rank) {
        if (mpi.rank == rank) {
//            std::cout << mpi.rank << " | rank indices = ";
            for (int j = 0; j < 25; ++j) {
                std::cout << Rank_indices[j] << ", ";
            }
            std::cout << std::endl;
        }
        MPI_Barrier(mpi.comm);
    }
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    std::vector<std::pair<long,long>> block_sizes = this->sorted_block_sizes();
    for (long i = mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        long block = block_sizes[i].first;
        this->_my_blocks[block] = true;
    }
    for (long i = 2 * mpi.num_processes - 1 - mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        long block = block_sizes[i].first;
        this->_my_blocks[block] = true;
    }
    this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, true);
}

void TwoHopBlockmodel::distribute_none_block_degree_balanced(const Graph &graph) {
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    std::vector<long> approximate_block_degrees;
    for (long i = 0; i < this->num_blocks; ++i) {
        approximate_block_degrees.push_back(this->_block_degrees[i]);
    }
    std::vector<long> sorted_indices = utils::argsort(approximate_block_degrees);
    for (long i = mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        long block = sorted_indices[i];
        this->_my_blocks[block] = true;
    }
    for (long i = 2 * mpi.num_processes - 1 - mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        long block = sorted_indices[i];
        this->_my_blocks[block] = true;
    }
//    std::vector<std::pair<long,long>> block_sizes = this->sorted_block_sizes();
//    for (long i = mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
//        long block = block_sizes[i].first;
//        this->_my_blocks[block] = true;
//    }
//    for (long i = 2 * mpi.num_processes - 1 - mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
//        long block = block_sizes[i].first;
//        this->_my_blocks[block] = true;
//    }
    this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, true);
}

void TwoHopBlockmodel::distribute_none_agg_block_degree_balanced(const Graph &graph) {
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    this->_my_vertices = utils::constant<long>(graph.num_vertices(), 0);
    std::vector<long> block_degrees = utils::constant<long>(graph.num_vertices(), 0);
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        long block = this->_block_assignment[vertex];
        block_degrees[vertex] = this->_block_degrees[block];
    }
    std::vector<long> sorted_indices = utils::argsort(block_degrees);
    for (long i = mpi.rank; i < graph.num_vertices(); i += 2 * mpi.num_processes) {
        long vertex = sorted_indices[i];
        this->_my_vertices[vertex] = 1;
    }
    for (long i = 2 * mpi.num_processes - 1 - mpi.rank; i < graph.num_vertices(); i += 2 * mpi.num_processes) {
        long vertex = sorted_indices[i];
        this->_my_vertices[vertex] = 1;
    }
    std::vector<std::pair<long,long>> block_sizes = this->sorted_block_sizes();
    for (long i = mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        long block = block_sizes[i].first;
        this->_my_blocks[block] = true;
    }
    for (long i = 2 * mpi.num_processes - 1 - mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        long block = block_sizes[i].first;
        this->_my_blocks[block] = true;
    }
    this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, true);
}

void TwoHopBlockmodel::distribute_2hop_round_robin(const NeighborList &neighbors) {
    // Step 1: decide which blocks to own
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    for (long i = mpi.rank; i < this->num_blocks; i += mpi.num_processes)
        this->_my_blocks[i] = true;
    // Step 2: find out which blocks are in the 2-hop radius of my blocks
    build_two_hop_blockmodel(neighbors);
}

void TwoHopBlockmodel::distribute_2hop_size_balanced(const NeighborList &neighbors) {
    // Step 1: decide which blocks to own
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    std::vector<std::pair<long,long>> block_sizes = this->sorted_block_sizes();
    for (long i = mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        long block = block_sizes[i].first;
        this->_my_blocks[block] = true;
    }
    for (long i = 2 * mpi.num_processes - 1 - mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        long block = block_sizes[i].first;
        this->_my_blocks[block] = true;
    }
    // Step 2: find out which blocks are in the 2-hop radius of my blocks
    build_two_hop_blockmodel(neighbors);
}

void TwoHopBlockmodel::distribute_2hop_snowball(const NeighborList &neighbors) {
    // Step 1: decide which blocks to own
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    // std::cout << "my vertices size: " << this->_my_vertices.size() << " neighbors size: " << neighbors.size() << std::endl;
    if (this->_my_vertices.size() == neighbors.size()) {  // if already done sampling, no need to do it again
        std::cout << "already done sampling, now just re-assigning blocks based on sampled vertices" << std::endl;
        for (size_t vertex = 0; vertex < neighbors.size(); ++vertex) {
            if (this->_my_vertices[vertex] == 0) continue;
            long block = this->_block_assignment[vertex];
            this->_my_blocks[block] = true;
        }
    } else {
        long target = ceil((double) neighbors.size() / (double) mpi.num_processes);
        this->_my_vertices = utils::constant<long>(neighbors.size(), 0);  // cannot send vector<bool>.data() over MPI
        std::unordered_set<long> frontier;
        // Snowball Sampling
        srand(mpi.num_processes + mpi.rank);
        long start = rand() % neighbors.size();  // replace this with a proper long distribution
        std::cout << "rank: " << mpi.rank << " with start = " << start << std::endl;
        this->_my_vertices[start] = 1;
        for (long neighbor : neighbors[start]) {
            frontier.insert(neighbor);
        }
        long block = this->_block_assignment[start];
        this->_my_blocks[block] = true;
        long num_vertices = 1;
        while (num_vertices < target) {
            std::unordered_set<long> new_frontier;
            for (long vertex : frontier) {
                if (this->_my_vertices[vertex] == 1) continue;
                this->_my_vertices[vertex] = 1;
                for (long neighbor : neighbors[vertex]) {
                    new_frontier.insert(neighbor);
                }
                block = this->_block_assignment[vertex];
                this->_my_blocks[block] = true;
                num_vertices++;
                if (num_vertices == target) break;
            }
            if (num_vertices < target && frontier.size() == 0) {  // restart with a new vertex that isn't already selected
                std::unordered_set<long> candidates;
                for (long i = 0; i < (long) neighbors.size(); ++i) {
                    if (this->_my_vertices[i] == 0) candidates.insert(i);
                }
                long index = rand() % candidates.size();
                auto it = candidates.begin();
                std::advance(it, index);
                start = *it;
                this->_my_vertices[start] = 1;
                for (long neighbor : neighbors[start]) {
                    new_frontier.insert(neighbor);
                }
                block = this->_block_assignment[start];
                this->_my_blocks[block] = true;
                num_vertices++;
            }
            frontier = std::unordered_set<long>(new_frontier);
        }
        // Some vertices may be unassigned across all ranks. Find out what they are, and assign 1/num_processes of them
        // to this process.
        std::vector<long> global_selected(neighbors.size(), 0);
        MPI_Allreduce(this->_my_vertices.data(), global_selected.data(), neighbors.size(), MPI_LONG, MPI_MAX, mpi.comm);
        // if (mpi.rank == 0) {
            // std::cout << "my selected: " << std::boolalpha;
            // utils::print<long>(this->_my_vertices);
            // std::cout << "globally selected: ";
            // utils::print<long>(global_selected);
        // }
        std::vector<long> vertices_left;
        for (long vertex = 0; vertex < (long) global_selected.size(); ++vertex) {
            if (global_selected[vertex] == 0) {
                vertices_left.push_back(vertex);
            }
        }
        // assign remaining vertices in round-robin fashion
        for (size_t i = mpi.rank; i < vertices_left.size(); i += mpi.num_processes) {
            long vertex = vertices_left[i];
            this->_my_vertices[vertex] = 1;
            block = this->_block_assignment[vertex];
            this->_my_blocks[block] = true;
        }
    }
    // Step 2: find out which blocks are in the 2-hop radius of my blocks
    this->build_two_hop_blockmodel(neighbors);
}

void TwoHopBlockmodel::initialize_edge_counts(const Graph &graph) {
    /// TODO: this recreates the matrix (possibly unnecessary)
    this->_num_nonempty_blocks = 0;
    std::shared_ptr<ISparseMatrix> blockmatrix;
    long num_buckets = graph.num_edges() / graph.num_vertices();
    if (args.transpose) {
        blockmatrix = std::make_shared<DictTransposeMatrix>(this->num_blocks, this->num_blocks, num_buckets);
    } else {
        blockmatrix = std::make_shared<DictMatrix>(this->num_blocks, this->num_blocks);
    }
    // This may or may not be faster with push_backs. TODO: test init & fill vs push_back
    std::vector<long> block_degrees_in = utils::constant<long>(this->num_blocks, 0);
    std::vector<long> block_degrees_out = utils::constant<long>(this->num_blocks, 0);
    std::vector<long> block_degrees = utils::constant<long>(this->num_blocks, 0);
    std::vector<long> block_sizes = utils::constant<long>(this->num_blocks, 0);
    std::vector<MapVector<long>> out_degree_histogram(this->num_blocks);
    std::vector<MapVector<long>> in_degree_histogram(this->num_blocks);
    // Initialize the blockmodel in parallel
    #pragma omp parallel default(none) \
    shared(blockmatrix, block_degrees_in, block_degrees_out, block_degrees, block_sizes, out_degree_histogram, in_degree_histogram, graph, args)
    {
        long tid = omp_get_thread_num();
        long num_threads = omp_get_num_threads();
        long my_num_blocks = ceil(double(this->num_blocks) / double(num_threads));
        long start = my_num_blocks * tid;
        long end = start + my_num_blocks;
        for (ulong vertex = 0; vertex < graph.num_vertices(); ++vertex) {
            long block = this->_block_assignment[vertex];
            if (block < start || block >= end || !this->_in_two_hop_radius[block])  // only modify blocks this thread is responsible for
                continue;
            if (block_sizes[block] == 0) {
                #pragma omp atomic
                this->_num_nonempty_blocks++;
            }
            block_sizes[block]++;
            out_degree_histogram[block][graph.out_neighbors(long(vertex)).size()]++;
            in_degree_histogram[block][graph.in_neighbors(long(vertex)).size()]++;
            for (long neighbor : graph.out_neighbors(long(vertex))) {
                long neighbor_block = this->_block_assignment[neighbor];
                if (!this->_in_two_hop_radius[neighbor_block]) {
                    continue;
                }
                long weight = 1;
                blockmatrix->add(block, neighbor_block, weight);
                block_degrees_out[block] += weight;
                block_degrees[block] += weight;
            }
            for (long neighbor : graph.in_neighbors(long(vertex))) {
                long neighbor_block = this->_block_assignment[neighbor];
                if (!this->_in_two_hop_radius[neighbor_block]) {
                    continue;
                }
                long weight = 1;
                if (args.transpose) {
                    std::shared_ptr<DictTransposeMatrix> blockmatrix_dtm =
                            std::dynamic_pointer_cast<DictTransposeMatrix>(blockmatrix);
                    blockmatrix_dtm->add_transpose(neighbor_block, block, weight);
                }
                block_degrees_in[block] += weight;
                if (block != neighbor_block) {
                    block_degrees[block] += weight;
                }
            }
        }
    }  // OMP_PARALLEL
    this->_blockmatrix = std::move(blockmatrix);
    this->_block_degrees_out = std::move(block_degrees_out);
    this->_block_degrees_in = std::move(block_degrees_in);
    this->_block_degrees = std::move(block_degrees);
    this->_block_sizes = std::move(block_sizes);
    this->_out_degree_histogram = std::move(out_degree_histogram);
    this->_in_degree_histogram = std::move(in_degree_histogram);
}

double TwoHopBlockmodel::log_posterior_probability() const {
    std::vector<long> my_blocks;
    if (args.distribute == "2hop-snowball" || args.distribute == "none-edge-balanced" ||
        args.distribute == "none-agg-block-degree-balanced") {
        my_blocks = utils::constant<long>(this->num_blocks, -1);
        for (long block = 0; block < this->num_blocks; ++block) {
            if (this->_my_blocks[block])
                my_blocks[block] = mpi.rank;
        }
        MPI_Allreduce(MPI_IN_PLACE, my_blocks.data(), this->num_blocks, MPI_LONG, MPI_MAX, mpi.comm);
        // utils::print<long>(my_blocks);
    }
    Indices nonzero_indices = this->_blockmatrix->nonzero();
    std::vector<double> all_values = utils::to_double<long>(this->_blockmatrix->values());
    std::vector<double> degrees_in;
    std::vector<double> degrees_out;
    std::vector<double> values;
    for (ulong i = 0; i < nonzero_indices.rows.size(); ++i) {
        long row = nonzero_indices.rows[i];
        if (args.distribute == "2hop-snowball") {
            if (my_blocks[row] != mpi.rank) continue;
        } else {
            if (this->_my_blocks[row] == false) continue;
        }
        // if (row % mpi.num_processes != mpi.rank) continue;
        values.push_back(all_values[i]);
        degrees_in.push_back(this->_block_degrees_in[nonzero_indices.cols[i]]);
        degrees_out.push_back(this->_block_degrees_out[nonzero_indices.rows[i]]);
    }
    std::vector<double> temp = values * utils::nat_log<double>(
        values / (degrees_out * degrees_in));
    double partial_sum = utils::sum<double>(temp);
    // MPI COMMUNICATION START
    double final_sum = 0.0;
    MPI_Allreduce(&partial_sum, &final_sum, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    // MPI COMMUNICATION END
    // Alternative Plan for sampled 2-hop blockmodel:
    // 1. Break all_values, degrees_in, degrees_out into row-like statuses
    // 2. Compute temp across the rows that you own
    // 3. Perform an AllReduce MAX to find missing values
    // 4. Sum across the rows to find the final_sum
    return final_sum;
}

bool TwoHopBlockmodel::owns_block(long block) const {
    return this->_my_blocks[block];
}

bool TwoHopBlockmodel::owns_vertex(long vertex) const {
    if (args.distribute == "2hop-snowball" || args.distribute == "none-edge-balanced" ||
        args.distribute == "none-agg-block-degree-balanced") {
        return this->_my_vertices[vertex];
    }
    long block = this->_block_assignment[vertex];
    return this->owns_block(block);
}

std::vector<std::pair<long,long>> TwoHopBlockmodel::sorted_block_sizes() const {
    std::vector<std::pair<long,long>> block_sizes;
    for (long i = 0; i < this->num_blocks; ++i) {
        block_sizes.emplace_back(i, 0);
    }
    for (const long &block : this->_block_assignment) {
        block_sizes[block].second++;
    }
    utils::radix_sort(block_sizes);
    return block_sizes;
//    std::sort(block_sizes.begin(), block_sizes.end(),
//              [](const std::pair<long, long> &a, const std::pair<long, long> &b) { return a.second > b.second; });
//    std::vector<long> block_sizes = utils::constant<long>(this->num_blocks, 0);
//    for (const long &block : this->_block_assignment) {
//        block_sizes[block]++;
//    }
//    std::vector<long> indices = utils::argsort(block_sizes);
//    std::vector<std::pair<long,long>> result;
//    for (long i = 0; i < this->num_blocks; ++i) {
//        result.emplace_back(indices[i], block_sizes[indices[i]]);
//    }
//    return result;
}

bool TwoHopBlockmodel::stores(long block) const {
    return this->_in_two_hop_radius[block];
}

bool TwoHopBlockmodel::validate(const Graph &graph) const {
    std::cout << "Validating..." << std::endl;
    std::vector<long> assignment(this->_block_assignment);
    Blockmodel correct(this->num_blocks, graph, this->block_reduction_rate, assignment);
    for (long row = 0; row < this->num_blocks; ++row) {
        for (long col = 0; col < this->num_blocks; ++col) {
            if (!(this->in_two_hop_radius()[row] || this->in_two_hop_radius()[col])) continue;
//            long this_val = this->blockmatrix()->get(row, col);
            long correct_val = correct.blockmatrix()->get(row, col);
            if (!this->blockmatrix()->validate(row, col, correct_val)) {
                std::cout << "ERROR::matrix[" << row << "," << col << "] is " << this->blockmatrix()->get(row, col) <<
                          " but should be " << correct_val << std::endl;
                return false;
            }
//            if (this_val != correct_val) return false;
        }
    }
    for (long block = 0; block < this->num_blocks; ++block) {
        bool valid = true;
        if (this->_block_degrees[block] != correct.degrees(block)) {
            std::cout << "ERROR::block degrees of " << block << " is " << this->_block_degrees[block] <<
                      " when it should be " << correct.degrees(block) << std::endl;
            valid = false;
        }
        if (this->_block_degrees_out[block] != correct.degrees_out(block)) {
            std::cout << "ERROR::block out-degrees of " << block << " is " << this->_block_degrees_out[block] <<
                      " when it should be " << correct.degrees_out(block) << std::endl;
            valid = false;
        }
        if (this->_block_degrees_in[block] != correct.degrees_in(block)) {
            std::cout << "ERROR::block in-degrees of " << block << " is " << this->_block_degrees_in[block] <<
                      " when it should be " << correct.degrees_in(block) << std::endl;
            valid = false;
        }
        if (!valid) {
            std::cout << "ERROR::error state | d_out: " << this->_block_degrees_out[block] << " d_in: " <<
                      this->_block_degrees_in[block] << " d: " << this->_block_degrees[block] <<
                      " self_edges: " << this->blockmatrix()->get(block, block) << std::endl;
            std::cout << "ERROR::correct state | d_out: " << correct.degrees_out(block) << " d_in: " <<
                      correct.degrees_in(block) << " d: " << correct.degrees(block) <<
                      " self_edges: " << correct.blockmatrix()->get(block, block) << std::endl;
            std::cout << "ERROR::Checking matrix for errors..." << std::endl;
            for (long row = 0; row < this->num_blocks; ++row) {
                for (long col = 0; col < this->num_blocks; ++col) {
                    //            long this_val = this->blockmatrix()->get(row, col);
                    long correct_val = correct.blockmatrix()->get(row, col);
                    if (!this->blockmatrix()->validate(row, col, correct_val)) {
                        std::cout << "matrix[" << row << "," << col << "] is " << this->blockmatrix()->get(row, col) <<
                                  " but should be " << correct_val << std::endl;
                        return false;
                    }
                    //            if (this_val != correct_val) return false;
                }
            }
            std::cout << "ERROR::Block degrees not valid, but no errors were found in matrix" << std::endl;
            return false;
        }
    }
    return true;
}
