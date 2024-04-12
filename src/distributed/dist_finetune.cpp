/**
 * The distributed finetuning phase of the stochastic block blockmodeling algorithm.
 */
#ifndef SBP_DIST_FINETUNE_HPP
#define SBP_DIST_FINETUNE_HPP

#include "distributed/dist_finetune.hpp"

#include "distributed/dist_common.hpp"
#include "entropy.hpp"
#include "finetune.hpp"

namespace finetune::dist {

std::vector<double> MCMC_RUNTIMES;
std::vector<unsigned long> MCMC_VERTEX_EDGES;
std::vector<long> MCMC_NUM_BLOCKS;
std::vector<unsigned long> MCMC_BLOCK_DEGREES;
std::vector<unsigned long long> MCMC_AGGREGATE_BLOCK_DEGREES;

const int MEMBERSHIP_T_BLOCK_LENGTHS[2] = {1, 1};
const MPI_Aint MEMBERSHIP_T_DISPLACEMENTS[2] = {0, sizeof(long)};
const MPI_Datatype MEMBERSHIP_T_TYPES[2] = {MPI_LONG, MPI_LONG};

//long MCMC_iterations = 0;
//double MCMC_time = 0.0;
//double MCMC_sequential_time = 0.0;
//double MCMC_parallel_time = 0.0;
//double MCMC_vertex_move_time = 0.0;
//ulong MCMC_moves = 0;
//long num_surrounded = 0;
std::ofstream my_file;

MPI_Datatype Membership_t;

std::vector<Membership> mpi_get_assignment_updates(const std::vector<Membership> &membership_updates) {
    int num_moves = (int) membership_updates.size();
    int rank_moves[mpi.num_processes];
//    double t1 = MPI_Wtime();
//    my_file << mpi.rank << "," << MCMC_iterations << "," << t1 - t0 << std::endl;
    MPI_Allgather(&num_moves, 1, MPI_INT, &rank_moves, 1, MPI_INT, mpi.comm);
    int offsets[mpi.num_processes];
    offsets[0] = 0;
    for (long i = 1; i < mpi.num_processes; ++i) {
        offsets[i] = offsets[i - 1] + rank_moves[i - 1];
    }
    long batch_vertex_moves = offsets[mpi.num_processes - 1] + rank_moves[mpi.num_processes - 1];
    std::vector<Membership> collected_membership_updates(batch_vertex_moves);
    MPI_Allgatherv(membership_updates.data(), num_moves, Membership_t, collected_membership_updates.data(),
                   rank_moves, offsets, Membership_t, mpi.comm);
    return collected_membership_updates;
}

void async_move(const Membership &membership, const Graph &graph, TwoHopBlockmodel &blockmodel) {
    EdgeWeights out_edges = edge_weights(graph.out_neighbors(), membership.vertex, false);
    EdgeWeights in_edges = edge_weights(graph.in_neighbors(), membership.vertex, true);
    Vertex v = { membership.vertex,
                 (long) graph.out_neighbors(membership.vertex).size(),
                 (long) graph.in_neighbors(membership.vertex).size() };
    VertexMove_v3 move {
            0.0, true, v, membership.block, out_edges, in_edges
    };
    blockmodel.move_vertex(move);
}

size_t update_blockmodel(const Graph &graph, TwoHopBlockmodel &blockmodel,
                         const std::vector<Membership> &membership_updates) {
    std::vector<Membership> collected_membership_updates = mpi_get_assignment_updates(membership_updates);
    for (const Membership &membership: collected_membership_updates) {
        if (membership.block == blockmodel.block_assignment(membership.vertex)) continue;
        async_move(membership, graph, blockmodel);
    }
    size_t vertex_moves = collected_membership_updates.size();
    return vertex_moves;
}

TwoHopBlockmodel &asynchronous_gibbs(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels) {
    MPI_Type_create_struct(2, MEMBERSHIP_T_BLOCK_LENGTHS, MEMBERSHIP_T_DISPLACEMENTS, MEMBERSHIP_T_TYPES, &Membership_t);
    MPI_Type_commit(&Membership_t);
    // MPI Datatype init
    if (blockmodel.getNum_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    long total_vertex_moves = 0;
    double old_entropy = args.nonparametric ?
            entropy::nonparametric::mdl(blockmodel, graph) :
            entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    blockmodel.setOverall_entropy(old_entropy);
    double new_entropy = 0;
    for (long iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        measure_imbalance_metrics(blockmodel, graph);
        double start_t = MPI_Wtime();
//        blockmodel.validate(graph);
        // Block assignment used to re-create the Blockmodel after each batch to improve mixing time of
        // asynchronous Gibbs sampling
        std::vector<long> block_assignment(blockmodel.block_assignment());
        std::vector<Membership> membership_updates = asynchronous_gibbs_iteration(blockmodel, graph);
        MCMC_RUNTIMES.push_back(MPI_Wtime() - start_t);
        // START MPI COMMUNICATION
        std::vector<Membership> collected_membership_updates = mpi_get_assignment_updates(membership_updates);
        // END MPI COMMUNICATION
        long vertex_moves = 0;
        for (const Membership &membership: collected_membership_updates) {
            if (block_assignment[membership.vertex] == membership.block) continue;
            vertex_moves++;
            async_move(membership, graph, blockmodel);
        }
//            blockmodel.validate(graph);
//            blockmodel.set_block_assignment(block_assignment);
//            blockmodel.build_two_hop_blockmodel(graph.out_neighbors());
//            blockmodel.initialize_edge_counts(graph);
        new_entropy = args.nonparametric ?
                entropy::nonparametric::mdl(blockmodel, graph) :
                entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
        double delta_entropy = new_entropy - old_entropy;
        old_entropy = new_entropy;
        delta_entropies.push_back(delta_entropy);
        if (mpi.rank == 0) {
            std::cout << "Itr: " << iteration << " vertex moves: " << vertex_moves << " delta S: "
                      << delta_entropy / new_entropy << std::endl;
        }
        total_vertex_moves += vertex_moves;
        MCMC_iterations++;
        // Early stopping
        if (early_stop(iteration, blockmodels.golden_ratio_not_reached(), new_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(new_entropy);
    if (mpi.rank == 0) std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    if (mpi.rank == 0) std::cout << blockmodel.getOverall_entropy() << std::endl;
    MPI_Type_free(&Membership_t);
    // are there more iterations with the 2-hop blockmodel due to restricted vertex moves?
    return blockmodel;
}

std::vector<Membership> asynchronous_gibbs_iteration(TwoHopBlockmodel &blockmodel, const Graph &graph,
                                                     const std::vector<long> &active_set, int batch) {
    std::vector<Membership> membership_updates;
    std::vector<long> vertices;
    if (active_set.empty())
        vertices = utils::range<long>(0, graph.num_vertices());
    else
        vertices = active_set;
    long batch_size = long(ceil(double(vertices.size()) / args.batches));
    long start = batch * batch_size;
    long end = std::min(long(vertices.size()), (batch + 1) * batch_size);
    #pragma omp parallel for schedule(dynamic) default(none) shared(start, end, vertices, blockmodel, graph, membership_updates)
    for (size_t index = start; index < end; ++index) {
        long vertex = vertices[index];
        if (!blockmodel.owns_vertex(vertex)) continue;
        VertexMove proposal = dist::propose_gibbs_move(blockmodel, vertex, graph);
        if (proposal.did_move) {
            #pragma omp critical (updates)
            {
                membership_updates.push_back(Membership{vertex, proposal.proposed_block});
            }
        }
    }
    return membership_updates;
}

bool early_stop(long iteration, bool golden_ratio_not_reached, double initial_entropy,
                std::vector<double> &delta_entropies) {
    size_t last_index = delta_entropies.size() - 1;
    if (delta_entropies[last_index] == 0.0) {
        return true;
    }
    if (iteration < 3) {
        return false;
    }
    double average = delta_entropies[last_index] + delta_entropies[last_index - 1] + delta_entropies[last_index - 2];
    average /= -3.0;
    double threshold;
    if (golden_ratio_not_reached) { // Golden ratio bracket not yet established
        threshold = 5e-4 * initial_entropy;
    } else {
        threshold = 1e-4 * initial_entropy;
    }
    return average < threshold;
}

Blockmodel &finetune_assignment(TwoHopBlockmodel &blockmodel, Graph &graph) {
    MPI_Type_create_struct(2, MEMBERSHIP_T_BLOCK_LENGTHS, MEMBERSHIP_T_DISPLACEMENTS, MEMBERSHIP_T_TYPES, &Membership_t);
    MPI_Type_commit(&Membership_t);
    if (mpi.rank == 0)
        std::cout << "Fine-tuning partition results after sample results have been extended to full graph" << std::endl;
    std::vector<double> delta_entropies;
    // TODO: Add number of finetuning iterations to evaluation
    long total_vertex_moves = 0;
//    double old_entropy = entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    double old_entropy = args.nonparametric ?
                  entropy::nonparametric::mdl(blockmodel, graph) :
                  entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    blockmodel.setOverall_entropy(old_entropy);
    double new_entropy = 0;
    for (long iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
//        std::cout << mpi.rank << " | starting iteration " << iteration << std::endl;
//        measure_imbalance_metrics(blockmodel, graph);
        double start_t = MPI_Wtime();
//        std::vector<long> block_assignment(blockmodel.block_assignment());
        std::vector<long> vertices = utils::range<long>(0, graph.num_vertices());
        size_t vertex_moves = 0;
        for (int batch = 0; batch < args.batches; ++batch) {
            std::vector<Membership> membership_updates = metropolis_hastings_iteration(blockmodel, graph);
            vertex_moves += update_blockmodel(graph, blockmodel, membership_updates);
        }
//        std::vector<Membership> membership_updates = metropolis_hastings_iteration(blockmodel, graph);
        MCMC_RUNTIMES.push_back(MPI_Wtime() - start_t);
        // TODO: [OPTIONAL] add option to skip this communication step
//        std::vector<Membership> collected_membership_updates = mpi_get_assignment_updates(membership_updates);
//        for (const Membership &membership: collected_membership_updates) {
//            if (membership.block == block_assignment[membership.vertex]) continue;
//            async_move(membership, graph, blockmodel);
//        }
//        size_t vertex_moves = collected_membership_updates.size();
//        new_entropy = entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
        new_entropy = args.nonparametric ?
                      entropy::nonparametric::mdl(blockmodel, graph) :
                      entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
        double delta_entropy = new_entropy - old_entropy;
        old_entropy = new_entropy;
        delta_entropies.push_back(delta_entropy);
        if (mpi.rank == 0) {
            std::cout << "Itr: " << iteration << " vertex moves: " << vertex_moves << " delta S: "
                      << delta_entropy / new_entropy << std::endl;
        }
//        std::cout << mpi.rank << " | finished computing entropy for iteration " << iteration << std::endl;
        total_vertex_moves += vertex_moves;
        MCMC_iterations++;
        // Early stopping
        if (early_stop(iteration, false, blockmodel.getOverall_entropy(), delta_entropies)) {
            std::cout << mpi.rank << " | this mpi rank early stopped at iteration " << iteration << " with dE = " << delta_entropy << std::endl;
            break;
        }
    }
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
    if (mpi.rank == 0) std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    if (mpi.rank == 0) std::cout << blockmodel.getOverall_entropy() << std::endl;
    MPI_Type_free(&Membership_t);
    return blockmodel;
}

TwoHopBlockmodel &hybrid_mcmc(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels) {
    MPI_Type_create_struct(2, MEMBERSHIP_T_BLOCK_LENGTHS, MEMBERSHIP_T_DISPLACEMENTS, MEMBERSHIP_T_TYPES, &Membership_t);
    MPI_Type_commit(&Membership_t);
    // MPI Datatype init
    if (blockmodel.getNum_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    size_t total_vertex_moves = 0;
    double old_entropy = args.nonparametric ?
            entropy::nonparametric::mdl(blockmodel, graph) :
            entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    blockmodel.setOverall_entropy(old_entropy);
    double new_entropy = 0;
    for (long iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        measure_imbalance_metrics(blockmodel, graph);
        double start_t = MPI_Wtime();
        std::vector<long> block_assignment(blockmodel.block_assignment());
        std::vector<Membership> membership_updates = metropolis_hastings_iteration(blockmodel, graph, graph.high_degree_vertices(), -1);
        size_t vertex_moves = update_blockmodel(graph, blockmodel, membership_updates);
        for (int batch = 0; batch < args.batches; ++batch) {
            std::vector<Membership> async_updates = asynchronous_gibbs_iteration(blockmodel, graph, graph.low_degree_vertices(), batch);
            vertex_moves += update_blockmodel(graph, blockmodel, async_updates);
        }
//        std::vector<Membership> async_updates = asynchronous_gibbs_iteration(blockmodel, graph, graph.low_degree_vertices());
//        utils::extend<Membership>(membership_updates, async_updates);
//        assert(membership_updates.size() >= async_updates.size());
        MCMC_RUNTIMES.push_back(MPI_Wtime() - start_t);
//        std::vector<Membership> collected_membership_updates = mpi_get_assignment_updates(membership_updates);
//        for (const Membership &membership: collected_membership_updates) {
//            if (membership.block == block_assignment[membership.vertex]) continue;
//            async_move(membership, graph, blockmodel);
//        }
//        size_t vertex_moves = collected_membership_updates.size();
        new_entropy = args.nonparametric ?
                entropy::nonparametric::mdl(blockmodel, graph) :
                entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
        double delta_entropy = new_entropy - old_entropy;
        old_entropy = new_entropy;
        delta_entropies.push_back(delta_entropy);
        if (mpi.rank == 0) {
            std::cout << "Itr: " << iteration << " vertex moves: " << vertex_moves << " delta S: "
                      << delta_entropy / new_entropy << std::endl;
        }
        total_vertex_moves += vertex_moves;
        MCMC_iterations++;
        if (early_stop(iteration, blockmodels.golden_ratio_not_reached(), new_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(new_entropy);
    if (mpi.rank == 0) std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    if (mpi.rank == 0) std::cout << blockmodel.getOverall_entropy() << std::endl;
    MPI_Type_free(&Membership_t);
    return blockmodel;
}

void measure_imbalance_metrics(const TwoHopBlockmodel &blockmodel, const Graph &graph) {
    std::vector<long> degrees = graph.degrees();
    MapVector<bool> block_count;
    unsigned long num_degrees = 0;
    unsigned long long num_aggregate_block_degrees = 0;
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        if (!blockmodel.owns_vertex(vertex)) continue;
        num_degrees += degrees[vertex];
        long block = blockmodel.block_assignment(vertex);
        block_count[block] = true;
        num_aggregate_block_degrees += blockmodel.degrees(block);
    }
    MCMC_VERTEX_EDGES.push_back(num_degrees);
    MCMC_NUM_BLOCKS.push_back(block_count.size());
    unsigned long block_degrees = 0;
    for (const std::pair<long, bool> &entry : block_count) {
        long block = entry.first;
        block_degrees += blockmodel.degrees(block);
    }
    MCMC_BLOCK_DEGREES.push_back(block_degrees);
    MCMC_AGGREGATE_BLOCK_DEGREES.push_back(num_aggregate_block_degrees);
}

TwoHopBlockmodel &metropolis_hastings(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels) {
    MPI_Type_create_struct(2, MEMBERSHIP_T_BLOCK_LENGTHS, MEMBERSHIP_T_DISPLACEMENTS, MEMBERSHIP_T_TYPES, &Membership_t);
    MPI_Type_commit(&Membership_t);
    // MPI Datatype init
    if (blockmodel.getNum_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    size_t total_vertex_moves = 0;
    double old_entropy = args.nonparametric ?
            entropy::nonparametric::mdl(blockmodel, graph) :
            entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    blockmodel.setOverall_entropy(old_entropy);
    double new_entropy = 0;
    for (long iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        measure_imbalance_metrics(blockmodel, graph);
        double start_t = MPI_Wtime();
        std::vector<long> block_assignment(blockmodel.block_assignment());
        std::vector<long> vertices = utils::range<long>(0, graph.num_vertices());
        size_t vertex_moves = 0;
        for (int batch = 0; batch < args.batches; ++batch) {
            if (mpi.rank == 0) std::cout << "communicating for batch = " << batch << std::endl;
            std::vector<Membership> membership_updates = metropolis_hastings_iteration(blockmodel, graph, vertices, batch);
            vertex_moves += update_blockmodel(graph, blockmodel, membership_updates);
        }
        MCMC_RUNTIMES.push_back(MPI_Wtime() - start_t);
        // TODO: [OPTIONAL] add option to skip this communication step
//        std::vector<Membership> collected_membership_updates = mpi_get_assignment_updates(membership_updates);
//        for (const Membership &membership: collected_membership_updates) {
//            if (membership.block == block_assignment[membership.vertex]) continue;
//            async_move(membership, graph, blockmodel);
//        }
//        size_t vertex_moves = collected_membership_updates.size();
        new_entropy = args.nonparametric ?
                entropy::nonparametric::mdl(blockmodel, graph) :
                entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
        double delta_entropy = new_entropy - old_entropy;
        old_entropy = new_entropy;
        delta_entropies.push_back(delta_entropy);
        if (mpi.rank == 0) {
            std::cout << "Itr: " << iteration << " vertex moves: " << vertex_moves << " delta S: "
                      << delta_entropy / new_entropy << std::endl;
        }
        total_vertex_moves += vertex_moves;
        MCMC_iterations++;
        if (early_stop(iteration, blockmodels.golden_ratio_not_reached(), new_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(new_entropy);
    if (mpi.rank == 0) {
        std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
        std::cout << blockmodel.getOverall_entropy() << std::endl;
    }
    MPI_Type_free(&Membership_t);
    return blockmodel;
}

std::vector<Membership> metropolis_hastings_iteration(TwoHopBlockmodel &blockmodel, Graph &graph,
                                                      const std::vector<long> &active_set, int batch) {
    std::vector<Membership> membership_updates;
    std::vector<long> vertices;
    if (active_set.empty())
        vertices = utils::range<long>(0, graph.num_vertices());
    else
        vertices = active_set;
    long batch_size = long(ceil(double(vertices.size()) / args.batches));
    long start = batch * batch_size;
    long end = std::min(long(vertices.size()), (batch + 1) * batch_size);
    // for hybrid_mcmc, we want to go through entire active_set in one go, regardless of number of batches
    if (batch == -1) {
        start = 0;
        end = (long) vertices.size();
    }
    for (size_t index = start; index < end; ++index) {  // long vertex : vertices) {
        long vertex = vertices[index];
        if (!blockmodel.owns_vertex(vertex)) continue;
        VertexMove proposal = dist::propose_mh_move(blockmodel, vertex, graph);
        if (proposal.did_move) {
            assert(blockmodel.stores(proposal.proposed_block));
            membership_updates.push_back(Membership{vertex, proposal.proposed_block});
        }
    }
    return membership_updates;
}

VertexMove propose_gibbs_move(const TwoHopBlockmodel &blockmodel, long vertex, const Graph &graph) {
    bool did_move = false;
    long current_block = blockmodel.block_assignment(vertex);
    if (blockmodel.block_size(current_block) == 1) {
        return VertexMove{0.0, did_move, -1, -1 };
    }

    EdgeWeights out_edges = edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = edge_weights(graph.in_neighbors(), vertex, true);

    utils::ProposalAndEdgeCounts proposal = common::dist::propose_new_block(
            current_block, out_edges, in_edges, blockmodel.block_assignment(), blockmodel, false);
    if (!blockmodel.stores(proposal.proposal)) {
        std::cerr << "ERROR " << "blockmodel doesn't own proposed block!!!!!" << std::endl;
        exit(-1000000000);
    }
    if (proposal.proposal == current_block) {
        return VertexMove{0.0, did_move, -1, -1 };
    }

    return eval_vertex_move(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
}

VertexMove propose_mh_move(TwoHopBlockmodel &blockmodel, long vertex, const Graph &graph) {
    bool did_move = false;
    long current_block = blockmodel.block_assignment(vertex);  // getBlock_assignment()[vertex];
    if (blockmodel.block_size(current_block) == 1) {
        return VertexMove{0.0, did_move, -1, -1 };
    }
    EdgeWeights out_edges = edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = edge_weights(graph.in_neighbors(), vertex, true);

//    blockmodel.validate(graph);
    utils::ProposalAndEdgeCounts proposal = common::dist::propose_new_block(
            current_block, out_edges, in_edges, blockmodel.block_assignment(), blockmodel, false);
    if (!blockmodel.stores(proposal.proposal)) {
        std::cerr << "ERROR " << "blockmodel doesn't own proposed block!!!!!" << std::endl;
        exit(-1000000000);
    }
    if (proposal.proposal == current_block) {
        return VertexMove{0.0, did_move, -1, -1 };
    }

    return move_vertex(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
}

}  // namespace finetune::dist

#endif // SBP_DIST_FINETUNE_HPP