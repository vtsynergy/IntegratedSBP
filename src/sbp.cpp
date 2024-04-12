#include "sbp.hpp"

#include "block_merge.hpp"
#include "entropy.hpp"
#include "finetune.hpp"
#include "fs.hpp"
#include "mpi_data.hpp"

#include "assert.h"
#include <fenv.h>
#include <sstream>

namespace sbp {

double total_time = 0.0;
long total_num_islands = 0;
double finetune_time = 0.0;

std::vector<intermediate> intermediate_results;

std::vector<intermediate> get_intermediates() {
    return intermediate_results;
}

/*void write_results(double iteration, std::ofstream &file, const Graph &graph, const Blockmodel &blockmodel, double mdl) {
    // fedisableexcept(FE_INVALID | FE_OVERFLOW);
    file << args.tag << "," << graph.num_vertices() << "," << args.overlap << "," << args.blocksizevar << ",";
    file << args.undirected << "," << args.algorithm << "," << iteration << ",";
    file << mdl << ","  << entropy::normalize_mdl_v1(mdl, graph.num_edges()) << ",";
    file << entropy::normalize_mdl_v2(mdl, graph.num_vertices(), graph.num_edges()) << ",";
    file << graph.modularity(blockmodel.block_assignment()) << "," << blockmodel.interblock_edges() << ",";
    file << blockmodel.block_size_variation() << std::endl;
    // feenableexcept(FE_INVALID | FE_OVERFLOW);
}*/

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
    intermediate.sort_time = Blockmodel_sort_time;
    intermediate.access_time = Blockmodel_access_time;
    intermediate.total_time = total_time;
    intermediate.update_assignment = Blockmodel_update_assignment;
    intermediate_results.push_back(intermediate);
    if (mpi.rank == 0)
        std::cout << "Iteration " << iteration << " MDL: " << mdl << " v1 normalized: " << normalized_mdl_v1
                  << " modularity: " << modularity << " MCMC iterations: " << finetune::MCMC_iterations << " MCMC time: "
                  << finetune::MCMC_time << " Block Merge time: " << block_merge::BlockMerge_time << " total time: "
                  << total_time << std::endl;
}

Blockmodel stochastic_block_partition(Graph &graph, Args &args, bool divide_and_conquer) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    Blockmodel blockmodel(graph.num_vertices(), graph, double(BLOCK_REDUCTION_RATE));
    common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
//    Blockmodel_first_build_time = BLOCKMODEL_BUILD_TIME;
    BLOCKMODEL_BUILD_TIME = 0.0;
    double initial_mdl = entropy::mdl(blockmodel, graph);
    add_intermediate(0, graph, -1, initial_mdl);
    BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
    double iteration = 0;
    while (!done_blockmodeling(blockmodel, blockmodel_triplet)) {
        if (divide_and_conquer) {
            if (!blockmodel_triplet.golden_ratio_not_reached() ||
                (blockmodel_triplet.get(0).getNum_blocks() > 1 && blockmodel_triplet.get(1).getNum_blocks() <= 1)) {
                blockmodel_triplet.status();
                blockmodel = blockmodel_triplet.get(0).copy();
                break;
            }
        }
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to " 
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        double start_bm = MPI_Wtime();
        blockmodel = block_merge::merge_blocks(blockmodel, graph, graph.num_edges());
        block_merge::BlockMerge_time += MPI_Wtime() - start_bm;
        if (iteration < 1) {
            double mdl = entropy::mdl(blockmodel, graph);
            add_intermediate(0.5, graph, -1, mdl);
        }
        std::cout << "Starting MCMC vertex moves" << std::endl;
        double start_mcmc = MPI_Wtime();
        common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
        if (args.algorithm == "async_gibbs" && iteration < double(args.asynciterations))
            blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
        else if (args.algorithm == "hybrid_mcmc")
            blockmodel = finetune::hybrid_mcmc(blockmodel, graph, blockmodel_triplet);
        else if (args.algorithm == "hybrid_mcmc_load_balanced")
            blockmodel = finetune::hybrid_mcmc_load_balanced(blockmodel, graph, blockmodel_triplet);
        else // args.algorithm == "metropolis_hastings"
            blockmodel = finetune::metropolis_hastings(blockmodel, graph, blockmodel_triplet);
        finetune::MCMC_time += MPI_Wtime() - start_mcmc;
        total_time += MPI_Wtime() - start_bm;
        add_intermediate(++iteration, graph, -1, blockmodel.getOverall_entropy());
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
        common::candidates = std::uniform_int_distribution<long>(0, blockmodel.getNum_blocks() - 2);
    }
    // only last iteration result will calculate expensive modularity
    double modularity = -1;
    if (args.modularity)
        modularity = graph.modularity(blockmodel.block_assignment());
    add_intermediate(-1, graph, modularity, blockmodel.getOverall_entropy());
    return blockmodel;
}

bool done_blockmodeling(Blockmodel &blockmodel, BlockmodelTriplet &blockmodel_triplet, long min_num_blocks) {
    if (min_num_blocks > 0) {
        if ((blockmodel.getNum_blocks() <= min_num_blocks) || !blockmodel_triplet.get(2).empty) {
            return true;
        }
    }
    if (blockmodel_triplet.optimal_num_blocks_found) {
        blockmodel_triplet.status();
        std::cout << "Optimal number of blocks was found" << std::endl;
        return true;
    }
    return false;
}

}  // namespace sbp
