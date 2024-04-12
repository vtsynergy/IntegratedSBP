#include "distributed/dist_sbp.hpp"

#include "distributed/dist_block_merge.hpp"
#include "distributed/dist_blockmodel_triplet.hpp"
#include "distributed/dist_finetune.hpp"
#include "entropy.hpp"
#include "finetune.hpp"
#include "distributed/two_hop_blockmodel.hpp"

#include <sstream>

namespace sbp::dist {

double total_time = 0.0;

double finetune_time = 0.0;

std::vector<intermediate> intermediate_results;

std::vector<intermediate> get_intermediates() {
    return intermediate_results;
}

void add_intermediate(double iteration, const Graph &graph, double modularity, double mdl) {
    double normalized_mdl_v1 = entropy::normalize_mdl_v1(mdl, graph);
//    double modularity = -1;
//    if (iteration == -1)
//        modularity = graph.modularity(blockmodel.block_assignment());
    intermediate intermediate {};
    intermediate.iteration = iteration;
    intermediate.mdl = mdl;
    intermediate.normalized_mdl_v1 = normalized_mdl_v1;
    intermediate.modularity = modularity;
    intermediate.mcmc_iterations = finetune::MCMC_iterations;
    intermediate.mcmc_time = finetune::MCMC_time;
    intermediate.mcmc_sequential_time = finetune::MCMC_sequential_time;
    intermediate.mcmc_parallel_time = finetune::MCMC_parallel_time;
    intermediate.mcmc_vertex_move_time = finetune::MCMC_vertex_move_time;
    intermediate.mcmc_moves = finetune::MCMC_moves;
    intermediate.block_merge_time = block_merge::BlockMerge_time;
    intermediate.block_merge_loop_time = block_merge::BlockMerge_loop_time;
    intermediate.blockmodel_build_time = BLOCKMODEL_BUILD_TIME;
    intermediate.finetune_time = finetune_time;
    intermediate.load_balancing_time = Load_balancing_time;
    intermediate.sort_time = Blockmodel_sort_time;
    intermediate.access_time = Blockmodel_access_time;
    intermediate.total_time = total_time;
    intermediate.update_assignment = Blockmodel_update_assignment;
    intermediate_results.push_back(intermediate);
    if (mpi.rank == 0) {
        std::cout << "Iteration " << iteration << " MDL: " << mdl << " v1 normalized: " << normalized_mdl_v1
                  << " modularity: " << modularity << " MCMC iterations: " << finetune::MCMC_iterations
                  << " MCMC time: "
                  << finetune::MCMC_time << " Block Merge time: " << block_merge::BlockMerge_time << " total time: "
                  << total_time << std::endl;
    }
}

void record_runtime_imbalance() {
    std::cout << "Recording runtime imbalance statistics" << std::endl;
    long recvcount = (long) finetune::dist::MCMC_RUNTIMES.size();
    std::cout << mpi.rank << " : recvcount = " << recvcount << " np = " << mpi.num_processes << std::endl;
    std::cout << mpi.rank << " : runtime[5] = " << finetune::dist::MCMC_RUNTIMES[5] << std::endl;
    std::cout << mpi.rank << " : runtimes size = " << finetune::dist::MCMC_RUNTIMES.size() << std::endl;
//    std::vector<double> all_mcmc_runtimes = utils::constant<double>(recvcount, 0);
    std::vector<double> all_mcmc_runtimes(recvcount * mpi.num_processes, 0.0);
    std::vector<unsigned long> all_mcmc_vertex_edges(recvcount * mpi.num_processes, 0);
    std::vector<long> all_mcmc_num_blocks(recvcount * mpi.num_processes, 0);
    std::vector<unsigned long> all_mcmc_block_degrees(recvcount * mpi.num_processes, 0);
    std::vector<unsigned long long> all_mcmc_aggregate_block_degrees(recvcount * mpi.num_processes, 0);
    std::cout << mpi.rank << " : allocated vector size = " << all_mcmc_runtimes.size() << std::endl;
    MPI_Gather(finetune::dist::MCMC_RUNTIMES.data(), recvcount, MPI_DOUBLE,
               all_mcmc_runtimes.data(), recvcount, MPI_DOUBLE, 0, mpi.comm);
    MPI_Gather(finetune::dist::MCMC_VERTEX_EDGES.data(), recvcount, MPI_UNSIGNED,
               all_mcmc_vertex_edges.data(), recvcount, MPI_UNSIGNED, 0, mpi.comm);
    MPI_Gather(finetune::dist::MCMC_NUM_BLOCKS.data(), recvcount, MPI_LONG,
               all_mcmc_num_blocks.data(), recvcount, MPI_LONG, 0, mpi.comm);
    MPI_Gather(finetune::dist::MCMC_BLOCK_DEGREES.data(), recvcount, MPI_UNSIGNED_LONG,
               all_mcmc_block_degrees.data(), recvcount, MPI_UNSIGNED_LONG, 0, mpi.comm);
    MPI_Gather(finetune::dist::MCMC_AGGREGATE_BLOCK_DEGREES.data(), recvcount, MPI_UNSIGNED_LONG_LONG,
               all_mcmc_aggregate_block_degrees.data(), recvcount, MPI_UNSIGNED_LONG_LONG, 0, mpi.comm);
    if (mpi.rank != 0) return;  // Only rank 0 should actually save a CSV file
    std::ostringstream filepath_stream;
    filepath_stream << args.csv << args.numvertices;
    std::string filepath_dir = filepath_stream.str();
    std::ostringstream filename_stream;
    filename_stream << args.csv << args.numvertices << "/" << args.type << "_" << mpi.num_processes
                    << "_ranks_imbalance.csv";
    std::string filepath = filename_stream.str();
    long attempt = 0;
    while (fs::exists(filepath)) {
        filename_stream = std::ostringstream();
        filename_stream << args.csv << args.numvertices << "/" << args.type << "_" << mpi.num_processes
                        << "_ranks_imbalance_" << attempt << ".csv";
        filepath = filename_stream.str();
        attempt++;
    }
    std::cout << std::boolalpha <<  "writing imbalance #s to " << filepath << std::endl;
    fs::create_directories(fs::path(filepath_dir));
    std::ofstream file;
    file.open(filepath, std::ios_base::app);
    file << "iteration, ";
    for (long j = 0; j < 5; ++j) {
        for (long i = 0; i < mpi.num_processes; ++i) {
            file << i;
            if (j == 4 && i == mpi.num_processes - 1) {
                file << std::endl;
            } else {
                file << ", ";
            }
        }
    }
    for (long iteration = 0; iteration < finetune::dist::MCMC_RUNTIMES.size(); ++iteration) {
        file << iteration << ", ";
        for (long rank = 0; rank < mpi.num_processes; ++rank) {
            size_t position = rank * finetune::dist::MCMC_RUNTIMES.size() + iteration;
            file << all_mcmc_runtimes[position] << ", ";
//            if (rank < mpi.num_processes - 1) file << ", ";
        }
        for (long rank = 0; rank < mpi.num_processes; ++rank) {
            size_t position = rank * finetune::dist::MCMC_RUNTIMES.size() + iteration;
            file << all_mcmc_vertex_edges[position] << ", ";
//            if (rank < mpi.num_processes - 1) file << ", ";
        }
        for (long rank = 0; rank < mpi.num_processes; ++rank) {
            size_t position = rank * finetune::dist::MCMC_RUNTIMES.size() + iteration;
            file << all_mcmc_num_blocks[position] << ", ";
//            if (rank < mpi.num_processes - 1) file << ", ";
        }
        for (long rank = 0; rank < mpi.num_processes; ++rank) {
            size_t position = rank * finetune::dist::MCMC_RUNTIMES.size() + iteration;
            file << all_mcmc_block_degrees[position] << ", ";
//            if (rank < mpi.num_processes - 1) file << ", ";
        }
        for (long rank = 0; rank < mpi.num_processes; ++rank) {
            size_t position = rank * finetune::dist::MCMC_RUNTIMES.size() + iteration;
            file << all_mcmc_aggregate_block_degrees[position];
            if (rank < mpi.num_processes - 1) file << ", ";
        }
        file << std::endl;
    }
    file.close();
}

// Blockmodel stochastic_block_partition(Graph &graph, MPI &mpi, Args &args) {
Blockmodel stochastic_block_partition(Graph &graph, Args &args, bool divide_and_conquer) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    if (mpi.rank == 0) std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    // DistBlockmodel blockmodel(graph, args, mpi);
    TwoHopBlockmodel blockmodel(graph.num_vertices(), graph, BLOCK_REDUCTION_RATE);
    common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
    // Blockmodel blockmodel(graph.num_vertices(), graph.out_neighbors(), BLOCK_REDUCTION_RATE);
    if (mpi.rank == 0)
        std::cout << "Performing stochastic block blockmodeling on graph with " << graph.num_vertices() << " vertices "
                  << " and " << blockmodel.getNum_blocks() << " blocks." << std::endl;
    DistBlockmodelTriplet blockmodel_triplet = DistBlockmodelTriplet();
    long iteration = 0;
    while (!dist::done_blockmodeling(blockmodel, blockmodel_triplet, 0)) {
        if (divide_and_conquer) {
            if (!blockmodel_triplet.golden_ratio_not_reached() ||
                (blockmodel_triplet.get(0).getNum_blocks() > 1 && blockmodel_triplet.get(1).getNum_blocks() <= 1)) {
//                MPI_Barrier(mpi.comm);
                blockmodel_triplet.status();
                blockmodel = blockmodel_triplet.get(0).copy();
                break;
            }
        }
        if (mpi.rank == 0 && blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to "
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = block_merge::dist::merge_blocks(blockmodel, graph);
        common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
        if (mpi.rank == 0) std::cout << "Starting MCMC vertex moves" << std::endl;
        if (args.algorithm == "async_gibbs" && iteration < args.asynciterations)
            blockmodel = finetune::dist::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
        else if (args.algorithm == "hybrid_mcmc" && iteration < args.asynciterations)
            blockmodel = finetune::dist::hybrid_mcmc(blockmodel, graph, blockmodel_triplet);
        else
            blockmodel = finetune::dist::metropolis_hastings(blockmodel, graph, blockmodel_triplet);
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
        common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
        iteration++;
    }
//    std::cout << "Total MCMC iterations: " << finetune::MCMC_iterations << std::endl;
    double modularity = -1;
    if (args.modularity)
        modularity = graph.modularity(blockmodel.block_assignment());
    add_intermediate(-1, graph, modularity, blockmodel.getOverall_entropy());
//    record_runtime_imbalance();
    return blockmodel;
}

bool done_blockmodeling(TwoHopBlockmodel &blockmodel, DistBlockmodelTriplet &blockmodel_triplet, long min_num_blocks) {
    if (mpi.rank == 0) std::cout << "distributed done_blockmodeling" << std::endl;
    if (min_num_blocks > 0) {
        if ((blockmodel.getNum_blocks() <= min_num_blocks) || !blockmodel_triplet.get(2).empty) {
            return true;
        }
    }
    if (blockmodel_triplet.optimal_num_blocks_found) {
        blockmodel_triplet.status();
        if (mpi.rank == 0) std::cout << "Optimal number of blocks was found" << std::endl;
        return true;
    }
    return false;
}

}  // namespace sbp::dist