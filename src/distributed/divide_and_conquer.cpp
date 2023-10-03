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
//
// Created by Frank on 3/23/2023.
//

#include "divide_and_conquer.hpp"

#include "block_merge.hpp"
#include "blockmodel_triplet.hpp"
#include "entropy.hpp"
#include "finetune.hpp"
#include "sbp.hpp"

#include <mpi.h>

double sample_time = 0.0;
double sample_extend_time = 0.0;
double finetune_time = 0.0;

namespace dnc {

std::vector<long> combine_partitions(const Graph &graph, long &offset, std::vector<std::vector<long>> &vertex_lists,
                                     std::vector<std::vector<long>> &assignment_lists) {
    // Iteratively merge blockmodels together until at most 4 are left
    while (vertex_lists.size() > 4) {  // Magic number = 4 taken from iHeartGraph code
        std::vector<std::vector<long>> new_rank_vertices;
        std::vector<std::vector<long>> new_rank_assignment;
        for (int piece = 0; piece < vertex_lists.size(); piece += 2) {
            if (piece == vertex_lists.size() - 1) {  // num pieces is odd, and this is last piece
                new_rank_vertices.push_back(vertex_lists[piece]);
                new_rank_assignment.push_back(assignment_lists[piece]);
                continue;
            }
            std::vector<long> combined_vertices = vertex_lists[piece];
            combined_vertices.reserve(combined_vertices.size() + vertex_lists[piece + 1].size());
            combined_vertices.insert(combined_vertices.end(), vertex_lists[piece + 1].begin(),
                                     vertex_lists[piece + 1].end());
            std::vector<long> combined_assignment = dnc::combine_two_blockmodels(
                    combined_vertices, assignment_lists[piece],assignment_lists[piece + 1], graph
            );
            new_rank_vertices.push_back(combined_vertices);
            new_rank_assignment.push_back(combined_assignment);
        }
        vertex_lists = std::move(new_rank_vertices);
        assignment_lists = std::move(new_rank_assignment);
    }
    // Merge remaining blockmodels together
    std::vector<long> combined_assignment = utils::constant<long>(graph.num_vertices(), -1);
    for (int piece = 0; piece < vertex_lists.size(); ++piece) {
        for (int index = 0; index < vertex_lists[piece].size(); ++index) {
            long vertex_index = vertex_lists[piece][index];
            long assignment = assignment_lists[piece][index] + offset;
            combined_assignment[vertex_index] = assignment;
        }
        offset += *std::max_element(assignment_lists[piece].begin(), assignment_lists[piece].end()) + 1;
    }
    return combined_assignment;
}

std::vector<long> combine_two_blockmodels(const std::vector<long> &combined_vertices,
                                          const std::vector<long> &assignment_a,
                                          const std::vector<long> &assignment_b, const Graph &original_graph) {
    std::vector<long> combined_mapping = utils::constant<long>(original_graph.num_vertices(), -1);
    for (int index = 0; index < combined_vertices.size(); ++index) {
        long true_vertex_index = combined_vertices[index];
        combined_mapping[true_vertex_index] = index;
    }
    std::vector<long> combined_assignment = assignment_a;
    long offset = *std::max_element(assignment_a.begin(), assignment_a.end()) + 1;
    combined_assignment.reserve(assignment_a.size() + assignment_b.size());
    for (const long &block : assignment_b) {
        combined_assignment.push_back(block + offset);
    }
    sample::Sample new_subgraph = sample::from_vertices(original_graph, combined_vertices, combined_mapping);
    long combined_num_blocks = *std::max_element(combined_assignment.begin(), combined_assignment.end()) + 1;
    Blockmodel blockmodel = Blockmodel(combined_num_blocks, new_subgraph.graph, 0.5, combined_assignment);
    Blockmodel merged_blockmodel = merge_blocks(blockmodel, new_subgraph, offset, combined_num_blocks);
    return merged_blockmodel.block_assignment();
}

void evaluate_partition(const Graph &graph, Blockmodel &blockmodel, double runtime) {
    if (mpi.rank != 0) return;
    evaluate::Eval result = evaluate::evaluate_blockmodel(graph, blockmodel);
    std::cout << "Final F1 score = " << result.f1_score << std::endl;
    std::cout << "Community detection runtime = " << runtime << "s" << std::endl;
    if (std::isnan(result.nmi) || std::isinf(result.nmi)) {
        result.nmi = 0.00;
    }
    write_results(graph, result, runtime);
}

Blockmodel finetune_partition(Blockmodel &blockmodel, const Graph &graph) {
    // Finetune final assignment
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges()));
    BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
    blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
    double iteration = sbp::get_intermediates().size();
    while (!sbp::done_blockmodeling(blockmodel, blockmodel_triplet)) {
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to "
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        double start_bm = MPI_Wtime();
        blockmodel = block_merge::merge_blocks(blockmodel, graph, graph.num_edges());
        block_merge::BlockMerge_time += MPI_Wtime() - start_bm;
        std::cout << "Starting MCMC vertex moves" << std::endl;
        double start_mcmc = MPI_Wtime();
        common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
        if (args.algorithm == "async_gibbs" && iteration < double(args.asynciterations))
            blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
        else if (args.algorithm == "hybrid_mcmc")
            blockmodel = finetune::hybrid_mcmc(blockmodel, graph, blockmodel_triplet);
        else // args.algorithm == "metropolis_hastings"
            blockmodel = finetune::metropolis_hastings(blockmodel, graph, blockmodel_triplet);
        finetune::MCMC_time += MPI_Wtime() - start_mcmc;
        sbp::total_time += MPI_Wtime() - start_bm;
        sbp::add_intermediate(++iteration, graph, -1, blockmodel.getOverall_entropy());
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
        common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
    }
    return blockmodel;
}

Blockmodel merge_blocks(const Blockmodel &blockmodel, const sample::Sample &subgraph, long my_num_blocks, long combined_num_blocks) {
    long partner_num_blocks = combined_num_blocks - my_num_blocks;
    std::vector<long> merge_from_blocks, merge_to_blocks;
    MapVector<std::pair<long, double>> best_merges;
    if (my_num_blocks < partner_num_blocks) {
        merge_from_blocks = utils::range<long>(0, my_num_blocks);
        merge_to_blocks = utils::range<long>(my_num_blocks, partner_num_blocks);
    } else {
        merge_from_blocks = utils::range<long>(my_num_blocks, partner_num_blocks);
        merge_to_blocks = utils::range<long>(0, my_num_blocks);
    }
    std::vector<long> block_map = utils::range<long>(0, blockmodel.getNum_blocks());
    for (long merge_from : merge_from_blocks) {
        best_merges[merge_from] = std::make_pair<long, double>(-1, std::numeric_limits<double>::max());
        for (long merge_to : merge_to_blocks) {
            // Calculate the delta entropy given the current block assignment
            EdgeWeights out_blocks = blockmodel.blockmatrix()->outgoing_edges(merge_from);
            EdgeWeights in_blocks = blockmodel.blockmatrix()->incoming_edges(merge_from);
            long k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
            long k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
            long k = k_out + k_in;
            utils::ProposalAndEdgeCounts proposal{merge_to, k_out, k_in, k};
            Delta delta = block_merge::blockmodel_delta(merge_from, proposal.proposal, blockmodel);
            long proposed_block_self_edges = blockmodel.blockmatrix()->get(merge_to, merge_to)
                                             + delta.get(merge_to, merge_to);
            double dE = entropy::block_merge_delta_mdl(merge_from, proposal, blockmodel, delta);
            if (dE < best_merges[merge_from].second) {
                best_merges[merge_from] = std::make_pair(merge_to, dE);
                block_map[merge_from] = merge_to;
            }
        }
    }
    std::vector<long> assignment = blockmodel.block_assignment();
    for (long i = 0; i < subgraph.graph.num_vertices(); ++i) {
        assignment[i] = block_map[assignment[i]];
    }
    std::vector<long> mapping = Blockmodel::build_mapping(assignment);
    for (size_t i = 0; i < assignment.size(); ++i) {
        long block = assignment[i];
        long new_block = mapping[block];
        assignment[i] = new_block;
    }
    return { (long) merge_to_blocks.size(), subgraph.graph, 0.5, assignment };
}

void receive_partition(int src, std::vector<std::vector<long>> &src_vertices,
                       std::vector<std::vector<long>> &src_assignments) {
    MPI_Status status;
    std::cout << "Root waiting for info from rank " << src << std::endl;
    int partner_num_vertices;
    MPI_Recv(&partner_num_vertices, 1, MPI_INT, src, NUM_VERTICES_TAG, MPI_COMM_WORLD, &status);
    std::vector<long> partner_vertices = utils::constant<long>(partner_num_vertices, -1);
    std::vector<long> partner_assignment = utils::constant<long>(partner_num_vertices, -1);
    MPI_Recv(partner_vertices.data(), partner_num_vertices, MPI_LONG, src, VERTICES_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(partner_assignment.data(), partner_num_vertices, MPI_LONG, src, BLOCKS_TAG, MPI_COMM_WORLD, &status);
    src_vertices.push_back(partner_vertices);
    src_assignments.push_back(partner_assignment);
    std::cout << "Root received info from rank " << src << std::endl;
}

void translate_local_partition(std::vector<long> &local_vertices, std::vector<long> &local_assignment,
                               const sample::Sample &subgraph, long num_vertices,
                               const std::vector<long> &partition_assignment) {
    local_vertices = utils::constant<long>(subgraph.graph.num_vertices(), -1);
    local_assignment = utils::constant<long>(subgraph.graph.num_vertices(), -1);
    #pragma omp parallel for schedule(dynamic) default(none) \
            shared(num_vertices, subgraph, partition_assignment, local_vertices, local_assignment)
    for (long vertex = 0; vertex < num_vertices; ++vertex) {
        long local_index = subgraph.mapping[vertex];
        if (local_index < 0) continue;  // vertex not present
        long assignment = partition_assignment[local_index];
        local_vertices[local_index] = vertex;
        local_assignment[local_index] = assignment;
    }
//    std::cout << "========= vertices from rank " << mpi.rank << " ===================" << std::endl;
//    utils::print<long>(local_vertices);
//    std::cout << "========= vertices from rank " << mpi.rank << " ===================" << std::endl;
}

void write_results(const Graph &graph, const evaluate::Eval &eval, double runtime) {
    std::vector<sbp::intermediate> intermediate_results;
    intermediate_results = sbp::get_intermediates();
    std::ostringstream filepath_stream;
    filepath_stream << args.csv << args.numvertices;
    std::string filepath_dir = filepath_stream.str();
    filepath_stream << "/" << args.type << ".csv";
    std::string filepath = filepath_stream.str();
    bool file_exists = fs::exists(filepath);
    std::cout << std::boolalpha <<  "writing results to " << filepath << " exists = " << file_exists << std::endl;
    fs::create_directories(fs::path(filepath_dir));
    std::ofstream file;
    file.open(filepath, std::ios_base::app);
    if (!file_exists) {
        file << "tag, numvertices, numedges, overlap, blocksizevar, undirected, algorithm, iteration, mdl, "
             << "normalized_mdl_v1, sample_size, modularity, f1_score, nmi, true_mdl, true_mdl_v1, sampling_algorithm, "
             << "runtime, sampling_time, sample_extend_time, finetune_time, mcmc_iterations, mcmc_time, "
             << "sequential_mcmc_time, parallel_mcmc_time, vertex_move_time, mcmc_moves, total_num_islands, "
             << "block_merge_time, block_merge_loop_time, blockmodel_build_time, finetune_time, "
             << "sort_time, load_balancing_time, access_time, update_assignment, total_time" << std::endl;
    }
    for (const sbp::intermediate &temp : intermediate_results) {
        file << args.tag << ", " << graph.num_vertices() << ", " << graph.num_edges() << ", " << args.overlap << ", "
             << args.blocksizevar << ", " << args.undirected << ", " << args.algorithm << ", " << temp.iteration << ", "
             << temp.mdl << ", " << temp.normalized_mdl_v1 << ", " << args.samplesize << ", "
             << temp.modularity << ", " << eval.f1_score << ", " << eval.nmi << ", " << eval.true_mdl << ", "
             << entropy::normalize_mdl_v1(eval.true_mdl, graph.num_edges()) << ", "
             << args.samplingalg << ", " << runtime << ", " << sample_time << ", " << sample_extend_time << ", "
             << finetune_time << ", " << temp.mcmc_iterations << ", " << temp.mcmc_time << ", "
             << temp.mcmc_sequential_time << ", " << temp.mcmc_parallel_time << ", "
             << temp.mcmc_vertex_move_time << ", " << temp.mcmc_moves << ", " << sbp::total_num_islands << ", "
             << temp.block_merge_time << ", " << temp.block_merge_loop_time << ", "
             << temp.blockmodel_build_time << ", " << temp.finetune_time << ", " << temp.sort_time << ", "
             << temp.load_balancing_time << ", " << temp.access_time << ", " << temp.update_assignment << ", "
             << temp.total_time << std::endl;
    }
    file.close();
}

}
