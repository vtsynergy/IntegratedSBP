#include "blockmodel.hpp"

#include "assert.h"
#include "mpi.h"
#include <queue>

#include "../args.hpp"
#include "mpi_data.hpp"
#include "typedefs.hpp"
#include "utils.hpp"

double BLOCKMODEL_BUILD_TIME = 0.0;
double Blockmodel_sort_time = 0.0;
double Blockmodel_access_time = 0.0;
double Blockmodel_update_assignment = 0.0;

double Blockmodel::block_size_variation() const {
    // Normalized using variance / max_variance, where max_variance = range^2 / 4
    // See: https://link.springer.com/content/pdf/10.1007/BF00143817.pdf
    std::vector<long> block_sizes(this->num_blocks, 0);
    for (long block : this->_block_assignment) {
        block_sizes[block]++;
    }
    double total = utils::sum<long>(block_sizes);
    double mean = total / double(this->num_blocks);
    double min = std::numeric_limits<double>::max(), max = std::numeric_limits<double>::min(), variance = 0;
    for (long block_size : block_sizes) {
        if (block_size < min) min = block_size;
        if (block_size > max) max = block_size;
        variance += double(block_size - mean) * double(block_size - mean);
    }
    variance /= double(this->num_blocks);
    double max_variance = (double(max - min) * double(max - mean)) / 4.0;
    return double(variance / max_variance);
}

std::vector<long> Blockmodel::build_mapping(const std::vector<long> &values) {
    std::map<long, bool> unique_map;
    for (size_t i = 0; i < values.size(); ++i) {
        unique_map[values[i]] = true;
    }
    std::vector<long> mapping = utils::constant<long>((long) values.size(), -1);
    long counter = 0;
    for (std::pair<long, bool> element : unique_map) {
        mapping[element.first] = counter;
        counter++;
    }
    return mapping;
}

double Blockmodel::difficulty_score() const {
    double norm_variance = this->block_size_variation();
    double longerblock_edges = this->interblock_edges();
    return (2.0 * norm_variance * longerblock_edges) / (norm_variance + longerblock_edges);
}

// TODO: move to block_merge.cpp
void Blockmodel::carry_out_best_merges(const std::vector<double> &delta_entropy_for_each_block,
                                       const std::vector<long> &best_merge_for_each_block) {
    std::cout << "=== dE range = " << *std::min_element(delta_entropy_for_each_block.begin(), delta_entropy_for_each_block.end()) << " to "
              << *std::max_element(delta_entropy_for_each_block.begin(), delta_entropy_for_each_block.end()) << std::endl;
    double sort_start_t = MPI_Wtime();
    /* typedef std::tuple<long, long, double> merge_t;
    auto cmp_fxn = [](merge_t left, merge_t right) { return std::get<2>(left) > std::get<2>(right); };
    std::cout << "building priority queue" << std::endl;
    std::priority_queue<merge_t, std::vector<merge_t>, decltype(cmp_fxn)> queue(cmp_fxn);
    for (long i = 0; i < (long) delta_entropy_for_each_block.size(); ++i) {
        queue.push(std::make_tuple(i, best_merge_for_each_block[i], delta_entropy_for_each_block[i]));
    }
    std::cout << "done building priority queue" << std::endl;*/
    std::vector<long> best_merges = utils::partial_sort_indices(delta_entropy_for_each_block,
                                                               this->num_blocks_to_merge + 1);
    double sort_end_t = MPI_Wtime();
    Blockmodel_sort_time += sort_end_t - sort_start_t;
    // std::vector<long> best_merges = utils::argsort(delta_entropy_for_each_block);
    std::vector<long> block_map = utils::range<long>(0, this->num_blocks);
    if (mpi.rank == 0) std::cout << "block map size: " << block_map.size() << std::endl;
    auto translate = [&block_map](long block) -> long {
        long b = block;
        do {
            if (b >= block_map.size())
                std::cout << "bm[" << b << "] = " << block_map[b] << std::endl;
            b = block_map[b];
        } while (block_map[b] != b);
        if (b >= block_map.size())
            std::cout << "final bm[" << block << "] = " << block_map[b] << std::endl;
        return b;
    };
    long num_merged = 0;
    long counter = 0;
    while (num_merged < this->num_blocks_to_merge) {
        /*merge_t best_merge = queue.top();
        queue.pop();
        long merge_from = std::get<0>(best_merge);
        long merge_to = block_map[std::get<1>(best_merge)];*/
        long merge_from = best_merges[counter];
        long merge_to = translate(best_merge_for_each_block[merge_from]);
        // long merge_to = block_map[best_merge_for_each_block[merge_from]];
        counter++;
        if (merge_to != merge_from) {
            block_map[merge_from] = merge_to;
            /*for (size_t i = 0; i < block_map.size(); ++i) {
                long block = block_map[i];
                if (block == merge_from) {
                    block_map[i] = merge_to;
                }
            }
            this->update_block_assignment(merge_from, merge_to);*/
            num_merged++;
        }
    }
    double update_start_t = MPI_Wtime();
    Blockmodel_access_time += update_start_t - sort_end_t;
    for (long i = 0; i < this->_block_assignment.size(); ++i) {
        this->_block_assignment[i] = translate(this->_block_assignment[i]);
    }
    Blockmodel_update_assignment += MPI_Wtime() - update_start_t;
    std::vector<long> mapping = build_mapping(this->_block_assignment);
    for (size_t i = 0; i < this->_block_assignment.size(); ++i) {
        long block = this->_block_assignment[i];
        long new_block = mapping[block];
        this->_block_assignment[i] = new_block;
    }
    this->num_blocks -= this->num_blocks_to_merge;
}

Blockmodel Blockmodel::clone_with_true_block_membership(const Graph &graph, std::vector<long> &true_block_membership) {
    long num_blocks = 0;
    std::vector<long> uniques = utils::constant<long>(true_block_membership.size(), 0);
    for (ulong i = 0; i < true_block_membership.size(); ++i) {
        long membership = true_block_membership[i];
        uniques[membership] = 1; // mark as used
    }
    for (ulong block = 0; block < uniques.size(); ++block) {
        if (uniques[block] == 1) {
            num_blocks++;
        }
    }
    return Blockmodel(num_blocks, graph, this->block_reduction_rate, true_block_membership);
}

Blockmodel Blockmodel::copy() {
    Blockmodel blockmodel_copy = Blockmodel(this->num_blocks, this->block_reduction_rate);
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
    blockmodel_copy.num_blocks_to_merge = 0;
    return blockmodel_copy;
}

Blockmodel Blockmodel::from_sample(long num_blocks, const Graph &graph, std::vector<long> &sample_block_membership,
                                 std::map<long, long> &mapping, double block_reduction_rate) {
    // Fill in initial block assignment
    std::vector<long> _block_assignment = utils::constant<long>(graph.num_vertices(), -1);  // neighbors.size(), -1);
    for (const auto &item : mapping) {
        _block_assignment[item.first] = sample_block_membership[item.second];
    }
    // Every unassigned block gets assigned to the next block number
    long next_block = num_blocks;
    for (ulong vertex = 0; vertex < graph.num_vertices(); ++vertex) {  // neighbors.size(); ++vertex) {
        if (_block_assignment[vertex] >= 0) {
            continue;
        }
        _block_assignment[vertex] = next_block;
        next_block++;
    }
    // Every previously unassigned block gets assigned to the block it's most connected to
    for (ulong vertex = 0; vertex < graph.num_vertices(); ++vertex) {  // neighbors.size(); ++vertex) {
        if (_block_assignment[vertex] < num_blocks) {
            continue;
        }
        std::vector<long> block_counts = utils::constant<long>(num_blocks, 0);
        // TODO: this can only handle unweighted graphs
        std::vector<long> vertex_neighbors = graph.out_neighbors(vertex);  // [vertex];
        for (ulong i = 0; i < vertex_neighbors.size(); ++i) {
            long neighbor = vertex_neighbors[i];
            long neighbor_block = _block_assignment[neighbor];
            if (neighbor_block < num_blocks) {
                block_counts[neighbor_block]++;
            }
        }
        long new_block = utils::argmax<long>(block_counts);
        // block_counts.maxCoeff(&new_block);
        _block_assignment[vertex] = new_block;
    }
    return Blockmodel(num_blocks, graph, block_reduction_rate, _block_assignment);
}

//void Blockmodel::initialize_edge_counts(const NeighborList &neighbors) {
//    double start = omp_get_wtime();
////    std::cout << "OLD BLOCKMODEL BOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO" << std::endl;
//    /// TODO: this recreates the matrix (possibly unnecessary)
//    if (args.transpose) {
//        this->_blockmatrix = std::make_shared<DictTransposeMatrix>(this->num_blocks, this->num_blocks);
//    } else {
//        this->_blockmatrix = std::make_shared<DictMatrix>(this->num_blocks, this->num_blocks);
//    }
//    // This may or may not be faster with push_backs. TODO: test init & fill vs push_back
//    this->_block_degrees_in = utils::constant<long>(this->num_blocks, 0);
//    this->_block_degrees_out = utils::constant<long>(this->num_blocks, 0);
//    this->_block_degrees = utils::constant<long>(this->num_blocks, 0);
//    // Initialize the blockmodel
//    // TODO: find a way to parallelize the matrix filling step
//    for (ulong vertex = 0; vertex < neighbors.size(); ++vertex) {
//        std::vector<long> vertex_neighbors = neighbors[vertex];
//        if (vertex_neighbors.empty()) {
//            continue;
//        }
//        long block = this->_block_assignment[vertex];
//        for (size_t i = 0; i < vertex_neighbors.size(); ++i) {
//            // Get count
//            long neighbor = vertex_neighbors[i];
//            long neighbor_block = this->_block_assignment[neighbor];
//            // TODO: change this once code is updated to support weighted graphs
//            long weight = 1;
//            // long weight = vertex_neighbors[i];
//            // Update blockmodel
//            this->_blockmatrix->add(block, neighbor_block, weight);
//            // Update degrees
//            this->_block_degrees_out[block] += weight;
//            this->_block_degrees_in[neighbor_block] += weight;
//            this->_block_degrees[block] += weight;
//            if (block != neighbor_block)
//                this->_block_degrees[neighbor_block] += weight;
//        }
//    }
//    double end = omp_get_wtime();
//    std::cout << omp_get_thread_num() << "Matrix creation walltime = " << end - start << std::endl;
//}

void Blockmodel::initialize_edge_counts(const Graph &graph) {  // Parallel version!
    this->_num_nonempty_blocks = 0;
    double build_start_t = MPI_Wtime();
    /// TODO: this recreates the matrix (possibly unnecessary)
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
    // Initialize the blockmodel
    #pragma omp parallel default(none) \
    shared(blockmatrix, block_degrees_in, block_degrees_out, block_degrees, block_sizes, out_degree_histogram, in_degree_histogram, graph, args)
    {
        long tid = omp_get_thread_num();
        long num_threads = omp_get_num_threads();
        long my_num_blocks = ceil(double(this->num_blocks) / double(num_threads));
        long start = my_num_blocks * tid;
        long end = start + my_num_blocks;
        for (ulong vertex = 0; vertex < graph.num_vertices(); ++vertex) {
//            std::vector<long> vertex_neighbors = graph.out_neighbors(vertex);  // neighbors[vertex];
//            if (vertex_neighbors.empty()) {
//                continue;
//            }
            long block = this->_block_assignment[vertex];
            if (block < start || block >= end)  // only modify blocks this thread is responsible for
                continue;
            /// TODO: in distributed version, may need to communicate this value at the end
            if (block_sizes[block] == 0) {
                #pragma omp atomic
                this->_num_nonempty_blocks++;
            }
            block_sizes[block]++;
            out_degree_histogram[block][graph.out_neighbors(long(vertex)).size()]++;
            in_degree_histogram[block][graph.in_neighbors(long(vertex)).size()]++;
            for (long neighbor : graph.out_neighbors(long(vertex))) {  // vertex_neighbors) {
//                size_t i = 0; i < vertex_neighbors.size(); ++i) {
                // Get count
//                long neighbor = vertex_neighbors[i];
                long neighbor_block = this->_block_assignment[neighbor];
                // TODO: change this once code is updated to support weighted graphs
                long weight = 1;
                // long weight = vertex_neighbors[i];
                // Update blockmodel
                blockmatrix->add(block, neighbor_block, weight);
                // Update degrees
                block_degrees_out[block] += weight;
                block_degrees[block] += weight;
//            }
            }
            for (long neighbor : graph.in_neighbors(long(vertex))) {
                long neighbor_block = this->_block_assignment[neighbor];
                long weight = 1;
                if (args.transpose) {
                    std::shared_ptr<DictTransposeMatrix> blockmatrix_dtm = std::dynamic_pointer_cast<DictTransposeMatrix>(blockmatrix);
                    blockmatrix_dtm->add_transpose(neighbor_block, block, weight);
                }
                // Update degrees
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
//    double end = omp_get_wtime();
//    std::cout << omp_get_thread_num() << "Matrix creation walltime = " << end - start << std::endl;
    BLOCKMODEL_BUILD_TIME += MPI_Wtime() - build_start_t;
}

double Blockmodel::interblock_edges() const {
    double num_edges = utils::sum<long>(this->_block_degrees_in);
    double interblock_edges = num_edges - double(this->_blockmatrix->trace());
    return interblock_edges / num_edges;
}

bool Blockmodel::is_neighbor_of(long block1, long block2) const {
    return this->blockmatrix()->get(block1, block2) + this->blockmatrix()->get(block2, block1) > 0;
}

double Blockmodel::log_posterior_probability() const {
    Indices nonzero_indices = this->_blockmatrix->nonzero();
    std::vector<double> values = utils::to_double<long>(this->_blockmatrix->values());
    std::vector<double> degrees_in;
    std::vector<double> degrees_out;
    for (ulong i = 0; i < nonzero_indices.rows.size(); ++i) {
        degrees_in.push_back(this->_block_degrees_in[nonzero_indices.cols[i]]);
        degrees_out.push_back(this->_block_degrees_out[nonzero_indices.rows[i]]);
    }
    for (ulong i = 0; i < values.size(); ++i) {
        if (degrees_in[i] == 0.0 || degrees_out[i] == 0.0) {
            std::cout << "value: " << values[i] << " degree_in: " << degrees_in[i] << " degree_out: " << degrees_out[i] << std::endl;
            exit(-1000);
        }
    }
    std::vector<double> temp = values * utils::nat_log<double>(
        values / (degrees_out * degrees_in));
    return utils::sum<double>(temp);
}

double Blockmodel::log_posterior_probability(long num_edges) const {
    if (args.undirected) {
        Indices nonzero_indices = this->_blockmatrix->nonzero();
        std::vector<double> values = utils::to_double<long>(this->_blockmatrix->values());
        std::vector<double> degrees_in;
        std::vector<double> degrees_out;
        for (ulong i = 0; i < nonzero_indices.rows.size(); ++i) {
            // This is OK bcause _block_degrees_in == _block_degrees_out == _block_degrees
            degrees_in.push_back(this->_block_degrees_in[nonzero_indices.cols[i]] / (2.0));
            degrees_out.push_back(this->_block_degrees_out[nonzero_indices.rows[i]] / (2.0));
        }
        std::vector<double> temp = values * utils::nat_log<double>(
            (values / (2.0)) / (degrees_out * degrees_in));
        double result = 0.5 * utils::sum<double>(temp);
        return result;
    }
    return log_posterior_probability();
}

void Blockmodel::update_block_assignment(long from_block, long to_block) {
    double start_t = MPI_Wtime();
    for (size_t index = 0; index < this->_block_assignment.size(); ++index) {
        if (this->_block_assignment[index] == from_block) {
            this->_block_assignment[index] = to_block;
        }
    }
    Blockmodel_update_assignment += MPI_Wtime() - start_t;
}

void Blockmodel::merge_block(long merge_from, long merge_to, const Delta &delta,
                             utils::ProposalAndEdgeCounts proposal) {
    long proposed_block_self_edges = this->blockmatrix()->get(merge_to, merge_to)
                                     + delta.get(merge_to, merge_to);
    this->update_block_assignment(merge_from, merge_to);
    // 2. Update the matrix
    this->blockmatrix()->update_edge_counts(delta);
//            blockmodel.degrees_out(new_block_degrees._block_degrees_out);
//            blockmodel.degrees_in(new_block_degrees._block_degrees_in);
//            blockmodel.degrees(new_block_degrees._block_degrees);
    this->degrees_out(merge_from, 0);
    this->degrees_out(merge_to, this->degrees_out(merge_to) + proposal.num_out_neighbor_edges);
    this->degrees_in(merge_from, 0);
    this->degrees_in(merge_to, this->degrees_in(merge_to) + proposal.num_in_neighbor_edges);
    this->degrees(merge_from, 0);
    this->degrees(merge_to, this->degrees_out(merge_to) + this->degrees_in(merge_to)
                                 - proposed_block_self_edges);
    this->_block_sizes[merge_to] += this->_block_sizes[merge_from];
    this->_block_sizes[merge_from] = 0;
    this->_num_nonempty_blocks--;
    for (const std::pair<long, long> &entry : this->_in_degree_histogram[merge_from]) {
        this->_in_degree_histogram[merge_to][entry.first] += entry.second;
    }
    for (const std::pair<long, long> &entry : this->_out_degree_histogram[merge_from]) {
        this->_out_degree_histogram[merge_to][entry.first] += entry.second;
    }
}

void Blockmodel::move_vertex(Vertex vertex, long current_block, long new_block, EdgeCountUpdates &updates,
                             std::vector<long> &new_block_degrees_out, std::vector<long> &new_block_degrees_in,
                             std::vector<long> &new_block_degrees) {
    this->_block_assignment[vertex.id] = new_block;
    this->update_edge_counts(current_block, new_block, updates);
    this->_block_degrees_out = new_block_degrees_out;
    this->_block_degrees_in = new_block_degrees_in;
    this->_block_degrees = new_block_degrees;
    this->_block_sizes[current_block]--;
    this->_block_sizes[new_block]++;
    if (this->_block_sizes[current_block] == 0) this->_num_nonempty_blocks--;
    if (this->_block_sizes[new_block] == 1) this->_num_nonempty_blocks++;
    
}

void Blockmodel::move_vertex(Vertex vertex, long current_block, long new_block, SparseEdgeCountUpdates &updates,
                             std::vector<long> &new_block_degrees_out, std::vector<long> &new_block_degrees_in,
                             std::vector<long> &new_block_degrees) {
    this->_block_assignment[vertex.id] = new_block;
    this->update_edge_counts(current_block, new_block, updates);
    this->_block_degrees_out = new_block_degrees_out;
    this->_block_degrees_in = new_block_degrees_in;
    this->_block_degrees = new_block_degrees;
    this->_block_sizes[current_block]--;
    this->_block_sizes[new_block]++;
    if (this->_block_sizes[current_block] == 0) this->_num_nonempty_blocks--;
    if (this->_block_sizes[new_block] == 1) this->_num_nonempty_blocks++;
}

void Blockmodel::move_vertex(Vertex vertex, long new_block, const Delta &delta,
                             std::vector<long> &new_block_degrees_out, std::vector<long> &new_block_degrees_in,
                             std::vector<long> &new_block_degrees) {
    long current_block = this->_block_assignment[vertex.id];
    this->_block_assignment[vertex.id] = new_block;
    this->_blockmatrix->update_edge_counts(delta);
    this->_block_degrees_out = new_block_degrees_out;
    this->_block_degrees_in = new_block_degrees_in;
    this->_block_degrees = new_block_degrees;
    this->_block_sizes[current_block]--;
    this->_block_sizes[new_block]++;
    if (this->_block_sizes[current_block] == 0) this->_num_nonempty_blocks--;
    if (this->_block_sizes[new_block] == 1) this->_num_nonempty_blocks++;
}

void Blockmodel::move_vertex(Vertex vertex, const Delta &delta, utils::ProposalAndEdgeCounts &proposal) {
//    std::cout << "vID = " << vertex.id << " proposal.proposal = " << proposal.proposal;
//    utils::print(this->_block_assignment);
    this->_block_assignment[vertex.id] = proposal.proposal;
//    utils::print(this->_block_assignment);
    this->_blockmatrix->update_edge_counts(delta);
    long current_block = delta.current_block();
    long current_block_self_edges = this->_blockmatrix->get(current_block, current_block);
//                                   + delta.get(current_block, current_block);
    long proposed_block_self_edges = this->_blockmatrix->get(proposal.proposal, proposal.proposal);
//                                    + delta.get(proposal.proposal, proposal.proposal);
    this->_block_degrees_out[current_block] -= proposal.num_out_neighbor_edges;
    this->_block_degrees_out[proposal.proposal] += proposal.num_out_neighbor_edges;
    this->_block_degrees_in[current_block] -= (proposal.num_in_neighbor_edges + delta.self_edge_weight());
    this->_block_degrees_in[proposal.proposal] += (proposal.num_in_neighbor_edges + delta.self_edge_weight());
    this->_block_degrees[current_block] = this->_block_degrees_out[current_block] +
            this->_block_degrees_in[current_block] - current_block_self_edges;
    this->_block_degrees[proposal.proposal] = this->_block_degrees_out[proposal.proposal] +
            this->_block_degrees_in[proposal.proposal] - proposed_block_self_edges;
    this->_block_sizes[current_block]--;
    this->_block_sizes[proposal.proposal]++;
    if (this->_block_sizes[current_block] == 0) this->_num_nonempty_blocks--;
    if (this->_block_sizes[proposal.proposal] == 1) this->_num_nonempty_blocks++;
    this->_out_degree_histogram[delta.current_block()][proposal.num_out_neighbor_edges]--;
    this->_in_degree_histogram[delta.current_block()][proposal.num_in_neighbor_edges]--;
    this->_out_degree_histogram[delta.proposed_block()][proposal.num_out_neighbor_edges]++;
    this->_in_degree_histogram[delta.proposed_block()][proposal.num_in_neighbor_edges]++;
}

void Blockmodel::move_vertex(const VertexMove_v3 &move) {
    long current_block = this->_block_assignment[move.vertex.id];
    for (const long &out_vertex : move.out_edges.indices) {  // Edge: vertex --> out_vertex
        long out_block = this->_block_assignment[out_vertex];
        this->_blockmatrix->sub(current_block, out_block, 1);
        this->_block_degrees_out[current_block]--;
        this->_block_degrees_out[move.proposed_block]++;
        if (out_vertex == move.vertex.id) {  // handle self edge
            this->_blockmatrix->add(move.proposed_block, move.proposed_block, 1);
            if (args.transpose) {
                std::shared_ptr<DictTransposeMatrix> blockmatrix_dtm =
                        std::dynamic_pointer_cast<DictTransposeMatrix>(this->_blockmatrix);
                blockmatrix_dtm->add_transpose(move.proposed_block, move.proposed_block, 1);
            }
            this->_block_degrees_in[current_block]--;
            this->_block_degrees_in[move.proposed_block]++;
        } else {
            this->_blockmatrix->add(move.proposed_block, out_block, 1);
            if (args.transpose) {
                std::shared_ptr<DictTransposeMatrix> blockmatrix_dtm =
                        std::dynamic_pointer_cast<DictTransposeMatrix>(this->_blockmatrix);
                blockmatrix_dtm->add_transpose(move.proposed_block, out_block, 1);
            }
        }
    }
    for (const long &in_vertex : move.in_edges.indices) {  // Edge: in_vertex --> vertex
        long in_block = this->_block_assignment[in_vertex];
        this->_blockmatrix->sub(in_block, current_block, 1);
        this->_block_degrees_in[current_block]--;
        this->_blockmatrix->add(in_block, move.proposed_block, 1);
        if (args.transpose) {
            std::shared_ptr<DictTransposeMatrix> blockmatrix_dtm = std::dynamic_pointer_cast<DictTransposeMatrix>(this->_blockmatrix);
            blockmatrix_dtm->add_transpose(in_block, move.proposed_block, 1);
        }
        this->_block_degrees_in[move.proposed_block]++;
    }
    this->_block_degrees[current_block] = this->_block_degrees_in[current_block] +
                                          this->_block_degrees_out[current_block] -
                                          this->_blockmatrix->get(current_block, current_block);
    this->_block_degrees[move.proposed_block] = this->_block_degrees_in[move.proposed_block] +
                                          this->_block_degrees_out[move.proposed_block] -
                                          this->_blockmatrix->get(move.proposed_block, move.proposed_block);
    this->_block_assignment[move.vertex.id] = move.proposed_block;
    this->_block_sizes[current_block]--;
    this->_block_sizes[move.proposed_block]++;
    if (this->_block_sizes[current_block] == 0) this->_num_nonempty_blocks--;
    if (this->_block_sizes[move.proposed_block] == 1) this->_num_nonempty_blocks++;
    this->_out_degree_histogram[current_block][move.vertex.out_degree]--;
    this->_in_degree_histogram[current_block][move.vertex.in_degree]--;
    this->_out_degree_histogram[move.proposed_block][move.vertex.out_degree]++;
    this->_in_degree_histogram[move.proposed_block][move.vertex.in_degree]++;
}

void Blockmodel::print_blockmatrix() const {
    this->_blockmatrix->print();
}

void Blockmodel::print_blockmodel() const {
    std::cout << "Blockmodel: " << std::endl;
    this->print_blockmatrix();
    std::cout << "Block degrees out: ";
    utils::print<long>(this->_block_degrees_out);
    std::cout << "Block degrees in: ";
    utils::print<long>(this->_block_degrees_in);
    std::cout << "Block degrees: ";
    utils::print<long>(this->_block_degrees);
    std::cout << "Assignment: ";
    utils::print<long>(this->_block_assignment);
    std::cout << "Block sizes: ";
    utils::print<long>(this->_block_sizes);
}

void Blockmodel::set_block_membership(long vertex, long block) { this->_block_assignment[vertex] = block; }

void Blockmodel::update_edge_counts(long current_block, long proposed_block, EdgeCountUpdates &updates) {
    this->_blockmatrix->update_edge_counts(current_block, proposed_block, updates.block_row, updates.proposal_row,
                                           updates.block_col, updates.proposal_col);
}

void Blockmodel::update_edge_counts(long current_block, long proposed_block, SparseEdgeCountUpdates &updates) {
    this->_blockmatrix->update_edge_counts(current_block, proposed_block, updates.block_row, updates.proposal_row,
                                           updates.block_col, updates.proposal_col);
}

bool Blockmodel::validate(const Graph &graph) const {
    std::cout << "Validating..." << std::endl;
    std::vector<long> assignment(this->_block_assignment);
    Blockmodel correct(this->num_blocks, graph, this->block_reduction_rate, assignment);
    for (long row = 0; row < this->num_blocks; ++row) {
        for (long col = 0; col < this->num_blocks; ++col) {
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
        if (this->_block_sizes[block] != correct.block_size(block)) {
            std::cout << "ERROR::block size of " << block << " is " << this->_block_sizes[block] <<
                      " when it should be " << correct.block_size(block) << std::endl;
            valid = false;
        }
        if (!valid) {
            std::cerr << "ERROR::error state | d_out: " << this->_block_degrees_out[block] << " d_in: " <<
                      this->_block_degrees_in[block] << " d: " << this->_block_degrees[block] <<
                      " self_edges: " << this->blockmatrix()->get(block, block) << " block size: " <<
                      this->_block_sizes[block] << std::endl;
            std::cerr << "ERROR::correct state | d_out: " << correct.degrees_out(block) << " d_in: " <<
                      correct.degrees_in(block) << " d: " << correct.degrees(block) <<
                      " self_edges: " << correct.blockmatrix()->get(block, block) << " block size: " <<
                      correct.block_size(block) << std::endl;
            std::cerr << "ERROR::Checking matrix for errors..." << std::endl;
            for (long row = 0; row < this->num_blocks; ++row) {
                for (long col = 0; col < this->num_blocks; ++col) {
        //            long this_val = this->blockmatrix()->get(row, col);
                    long correct_val = correct.blockmatrix()->get(row, col);
                    if (!this->blockmatrix()->validate(row, col, correct_val)) {
                        std::cerr << "matrix[" << row << "," << col << "] is " << this->blockmatrix()->get(row, col) <<
                        " but should be " << correct_val << std::endl;
                        return false;
                    }
        //            if (this_val != correct_val) return false;
                }
            }
            std::cerr << "ERROR::Block degrees not valid, but no errors were found in matrix" << std::endl;
            return false;
        }
    }
    if (this->_in_degree_histogram.size() != correct._in_degree_histogram.size()) {
        std::cerr << "ERROR::in degree histogram sizes don't match: " << this->_in_degree_histogram.size() << " != "
        << correct._in_degree_histogram.size() << std::endl;
        return false;
    }
    if (this->_out_degree_histogram.size() != correct._out_degree_histogram.size()) {
        std::cerr << "ERROR::out degree histogram sizes don't match: " << this->_out_degree_histogram.size() << " != "
                  << correct._out_degree_histogram.size() << std::endl;
        return false;
    }
    for (long block = 0; block < this->num_blocks; ++block) {
        if (this->_in_degree_histogram[block].size() != correct._in_degree_histogram[block].size()) {
            std::cerr << "ERROR::in degree histogram[" << block << "] sizes don't match: "
                      << this->_in_degree_histogram[block].size() << " != "
                      << correct._in_degree_histogram[block].size() << std::endl;
            return false;
        }
        if (this->_out_degree_histogram[block].size() != correct._out_degree_histogram[block].size()) {
            std::cerr << "ERROR::out degree histogram[" << block << "] sizes don't match: "
                      << this->_out_degree_histogram[block].size() << " != "
                      << correct._out_degree_histogram[block].size() << std::endl;
            return false;
        }
        for (const std::pair<long, long> &bar : this->_in_degree_histogram[block]) {
            if (bar.second != correct._in_degree_histogram[block][bar.first]) {
                std::cerr << "ERROR: in_degree histogram does not match at [" << block << "][" << bar.first << "]: "
                          << "expected value = " << correct._in_degree_histogram[block][bar.first] << " but got: "
                          << bar.second << std::endl;
                return false;
            }
        }
        for (const std::pair<long, long> &bar : this->_out_degree_histogram[block]) {
            if (bar.second != correct._out_degree_histogram[block][bar.first]) {
                std::cerr << "ERROR: out_degree histogram does not match at [" << block << "][" << bar.first << "]: "
                          << "expected value = " << correct._out_degree_histogram[block][bar.first] << " but got: "
                          << bar.second << std::endl;
                return false;
            }
        }
    }
    return true;
}
