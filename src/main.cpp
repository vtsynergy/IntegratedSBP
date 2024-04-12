
#include <chrono>
//#include <execinfo.h>
//#include <fenv.h>  // break on nans or infs
#include <iostream>
#include <mpi.h>
//#include <signal.h>
#include <string>

#include <nlohmann/json.hpp>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "distributed/dist_sbp.hpp"
#include "distributed/dist_finetune.hpp"
#include "entropy.hpp"
#include "evaluate.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "mpi_data.hpp"
#include "partition.hpp"
#include "rng.hpp"
#include "sample.hpp"
#include "sbp.hpp"


MPI_t mpi;
Args args;

double sample_time = 0.0;
double sample_extend_time = 0.0;
double finetune_time = 0.0;

struct Partition {
    Graph graph;
    Blockmodel blockmodel;
};

void write_json(const Blockmodel &blockmodel, double runtime) {
    nlohmann::json output;
    output["Runtime (s)"] = runtime;
    output["Filepath"] = args.filepath;
    output["Tag"] = args.tag;
    output["Algorithm"] = args.algorithm;
    output["Degree Product Sort"] = args.degreeproductsort;
    output["Data Distribution"] = args.distribute;
    output["Greedy"] = args.greedy;
    output["Metropolis-Hastings Ratio"] = args.mh_percent;
    output["Overlap"] = args.overlap;
    output["Block Size Variation"] = args.blocksizevar;
    output["Sample Size"] = args.samplesize;
    output["Sampling Algorithm"] = args.samplingalg;
    output["Num. Subgraphs"] = args.subgraphs;
    output["Subgraph Partition"] = args.subgraphpartition;
    output["Num. Threads"] = args.threads;
    output["Num. Processes"] = mpi.num_processes;
    output["Type"] = args.type;
    output["Undirected"] = args.undirected;
    output["Num. Vertex Moves"] = finetune::MCMC_moves;
    output["Num. MCMC Iterations"] = finetune::MCMC_iterations;
    output["Results"] = blockmodel.block_assignment();
    output["Description Length"] = blockmodel.getOverall_entropy();
    fs::create_directories(fs::path(args.json));
    std::ostringstream output_filepath_stream;
    output_filepath_stream << args.json << "/" << args.output_file;
    std::string output_filepath = output_filepath_stream.str();
    std::cout << "Saving results to file: " << output_filepath << std::endl;
    std::ofstream output_file;
    output_file.open(output_filepath, std::ios_base::app);
    output_file << std::setw(4) << output << std::endl;
    output_file.close();
}

void write_results(const Graph &graph, const evaluate::Eval &eval, double runtime) {
    std::vector<sbp::intermediate> intermediate_results;
    if (mpi.num_processes > 1) {
        intermediate_results = sbp::dist::get_intermediates();
    } else {
        intermediate_results = sbp::get_intermediates();
    }
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
             << "sort_time, load_balancing_time, access_time, update_assignmnet, total_time" << std::endl;
    }
    for (const sbp::intermediate &temp : intermediate_results) {
        file << args.tag << ", " << graph.num_vertices() << ", " << graph.num_edges() << ", " << args.overlap << ", "
             << args.blocksizevar << ", " << args.undirected << ", " << args.algorithm << ", " << temp.iteration << ", "
             << temp.mdl << ", " << temp.normalized_mdl_v1 << ", " << args.samplesize << ", "
             << temp.modularity << ", " << eval.f1_score << ", " << eval.nmi << ", " << eval.true_mdl << ", "
             << entropy::normalize_mdl_v1(eval.true_mdl, graph) << ", "
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

void evaluate_partition(Graph &graph, Blockmodel &blockmodel, double runtime) {
    if (mpi.rank != 0) return;
    write_json(blockmodel, runtime);
    if (!args.evaluate) return;
    evaluate::Eval result = evaluate::evaluate_blockmodel(graph, blockmodel);
    std::cout << "Final F1 score = " << result.f1_score << std::endl;
    std::cout << "Community detection runtime = " << runtime << "s" << std::endl;
    if (std::isnan(result.nmi) || std::isinf(result.nmi)) {
        result.nmi = 0.00;
    }
    write_results(graph, result, runtime);
}

void run(Partition &partition) {
    sbp::total_num_islands = partition.graph.num_islands();
    if (mpi.num_processes > 1) {
        partition.blockmodel = sbp::dist::stochastic_block_partition(partition.graph, args);
    } else {
        partition.blockmodel = sbp::stochastic_block_partition(partition.graph, args);
    }
}

int main(int argc, char* argv[]) {
    // signal(SIGABRT, handler);
    // long rank, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(mpi.comm, &mpi.rank);
    MPI_Comm_size(mpi.comm, &mpi.num_processes);
    // std::cout << "rank: " << mpi.rank << " np: " << mpi.num_processes << std::endl;

    args = Args(argc, argv);
    rng::init_generators();

    if (mpi.rank == 0) {
        std::cout << "Number of processes = " << mpi.num_processes << std::endl;
        // std::cout << "Parsed out the arguments" << std::endl;
    }
    // TODO: figure out how to distribute the graph if it doesn't fit in memory
    Graph graph = Graph::load();
    sample::Sample detached;
    Partition partition;
    double start = MPI_Wtime();
    if (args.detach) {  // if we're getting rid of vertices with degree < 2
        detached = sample::detach(graph);
        partition.graph = std::move(detached.graph);
        std::cout << "detached num vertices: " << partition.graph.num_vertices() << " E: "
                  << partition.graph.num_edges() << std::endl;
    } else {
        partition.graph = std::move(graph);
    }
    if (args.samplesize <= 0.0) {
        std::cerr << "ERROR " << "Sample size of " << args.samplesize << " is too low. Must be greater than 0.0" << std::endl;
        exit(-5);
    }
    if (args.samplesize < 1.0) {
        double sample_start_t = MPI_Wtime();
        if (mpi.rank == 0) std::cout << "Running sampling with size: " << args.samplesize << std::endl;
//        sample::Sample s = sample::max_degree(partition.graph);
        sample::Sample s = sample::sample(partition.graph);
        if (mpi.num_processes > 1) {
            MPI_Bcast(s.mapping.data(), (int) partition.graph.num_vertices(), MPI_LONG, 0, mpi.comm);
            if (mpi.rank > 0) {
                std::vector<long> vertices;
                for (const long &mapped_id : s.mapping) {
                    if (mapped_id >= 0)
                        vertices.push_back(mapped_id);
                }
                s = sample::from_vertices(partition.graph, vertices, s.mapping);
            }
            MPI_Barrier(mpi.comm);
        }
        Partition sample_partition;
        sample_partition.graph = std::move(s.graph);  // s.graph may be empty now
        // add timer
        double sample_end_t = MPI_Wtime();
        sample_time = sample_end_t - sample_start_t;
        run(sample_partition);
        double extend_start_t = MPI_Wtime();
        s.graph = std::move(sample_partition.graph);  // refill s.graph
        // extend sample to full graph
        // TODO: this seems deterministic...
        std::vector<long> assignment = sample::extend(partition.graph, sample_partition.blockmodel, s);
        // fine-tune full graph
        double finetune_start_t = MPI_Wtime();
        if (mpi.num_processes > 1) {
            // make sure the assignment is the same across processes
            MPI_Bcast(assignment.data(), (int) assignment.size(), MPI_LONG, 0, mpi.comm);
            Rank_indices = std::vector<long>();  // reset the rank_indices
            auto blockmodel = TwoHopBlockmodel(sample_partition.blockmodel.getNum_blocks(), partition.graph, 0.5, assignment);
            partition.blockmodel = finetune::dist::finetune_assignment(blockmodel, partition.graph);
        } else {
            partition.blockmodel = Blockmodel(sample_partition.blockmodel.getNum_blocks(), partition.graph, 0.5, assignment);
            partition.blockmodel = finetune::finetune_assignment(partition.blockmodel, partition.graph);
        }
        double finetune_end_t = MPI_Wtime();
        sample_extend_time = finetune_start_t - extend_start_t;
        finetune_time = finetune_end_t - finetune_start_t;
    } else {
        std::cout << "Running without sampling." << std::endl;
        run(partition);
    }
    if (args.detach) {
        std::cout << "Reattaching island and 1-degree vertices" << std::endl;
        partition.blockmodel = sample::reattach(graph, partition.blockmodel, detached);
    } else {
        graph = std::move(partition.graph);
    }
    // evaluate
    double end = MPI_Wtime();
    if (mpi.rank == 0) evaluate_partition(graph, partition.blockmodel, end - start);

    MPI_Finalize();
}
