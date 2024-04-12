#include "distributed/dist_block_merge.hpp"

namespace block_merge::dist {

MPI_Datatype Merge_t;

std::vector<Merge> mpi_get_best_merges(std::vector<Merge> &merge_buffer, int my_blocks) {
    // Get list of best merges owned by this MPI rank. Used in Allgatherv.
    std::vector<Merge> best_merges(my_blocks);
    long index = 0;
    for (const Merge &merge: merge_buffer) {
        if (merge.block == -1) continue;
        best_merges[index] = merge;
        index++;
    }
    int numblocks[mpi.num_processes];
    MPI_Allgather(&(my_blocks), 1, MPI_INT, &numblocks, 1, MPI_INT, mpi.comm);
    int offsets[mpi.num_processes];
    offsets[0] = 0;
    for (long i = 1; i < mpi.num_processes; ++i) {
        offsets[i] = offsets[i - 1] + numblocks[i - 1];
    }
    long total_blocks = offsets[mpi.num_processes - 1] + numblocks[mpi.num_processes - 1];
    // TODO: change the size of this to total_blocks? Otherwise when there is overlapping computation there may be a segfault
    std::vector<Merge> all_best_merges(total_blocks);
    MPI_Allgatherv(best_merges.data(), my_blocks, Merge_t, all_best_merges.data(), numblocks, offsets,
                   Merge_t, mpi.comm);
    return all_best_merges;
}

TwoHopBlockmodel &merge_blocks(TwoHopBlockmodel &blockmodel, const Graph &graph) {
    // MPI Datatype init
    int merge_blocklengths[3] = {1, 1, 1};
    MPI_Aint merge_displacements[3] = {0, sizeof(long), sizeof(long) + sizeof(long)};
    MPI_Datatype merge_types[3] = {MPI_LONG, MPI_LONG, MPI_DOUBLE};
    MPI_Type_create_struct(3, merge_blocklengths, merge_displacements, merge_types, &Merge_t);
    MPI_Type_commit(&Merge_t);
    // MPI Datatype init
    long num_blocks = blockmodel.getNum_blocks();
    std::vector<long> block_assignment = utils::range<long>(0, num_blocks);
    // long my_blocks = ceil(((double) num_blocks - (double) mpi.rank) / (double) mpi.num_processes);
    // merge_buffer stores best Merges as if all blocks are owned by this MPI rank. Used to avoid locks
    std::vector<Merge> merge_buffer(num_blocks);
    // long num_avoided = 0;  // number of avoided/skipped calculations
    // long index = 0;
    long my_blocks = 0;
    #pragma omp parallel for schedule(dynamic) default(none) \
    shared(num_blocks, blockmodel, my_blocks, graph, block_assignment, merge_buffer) // reduction( + : num_avoided)
    for (long current_block = 0; current_block < num_blocks; ++current_block) {
        // for (long current_block = mpi.rank; current_block < num_blocks; current_block += mpi.num_processes) {
        if (!blockmodel.owns_block(current_block)) continue;
        #pragma omp atomic update
        my_blocks++;
        std::unordered_map<long, bool> past_proposals;
        for (long i = 0; i < NUM_AGG_PROPOSALS_PER_BLOCK; ++i) {
            ProposalEvaluation proposal = propose_merge_sparse(current_block, blockmodel, graph,past_proposals);
            // std::cout << "proposal = " << proposal.proposed_block << " with DE " << proposal.delta_entropy << std::endl;
            // TODO: find a way to do this without having a large merge buffer. Maybe store list of my blocks in
            // TwoHopBlockmodel?
            if (proposal.delta_entropy < merge_buffer[current_block].delta_entropy) {
                merge_buffer[current_block] = Merge{current_block, proposal.proposed_block,
                                                    proposal.delta_entropy};
            }
        }
    }
    // MPI COMMUNICATION
    std::vector<Merge> all_best_merges = mpi_get_best_merges(merge_buffer, my_blocks);
    // END MPI COMMUNICATION
    std::vector<long> best_merge_for_each_block = utils::constant<long>(num_blocks, -1);
    std::vector<double> delta_entropy_for_each_block = utils::constant<double>(num_blocks, -1);
    // TODO: use a more intelligent way to assign these when there is overlap?
    for (const Merge &m: all_best_merges) {
//        std::cout << "block: " << m.block << " proposal: " << m.proposal << " dE: " << m.delta_entropy << std::endl;
//        if (mpi.rank == 0) std::cout << "rank " << mpi.rank << " | m.block = " << m.block << " num blocks = " << best_merge_for_each_block.size() << std::endl;
        best_merge_for_each_block[m.block] = m.proposal;
        delta_entropy_for_each_block[m.block] = m.delta_entropy;
    }
    // std::cout << mpi.rank << " best merges";
    // utils::print<long>(best_merge_for_each_block);
    if (args.approximate)
        blockmodel.carry_out_best_merges(delta_entropy_for_each_block, best_merge_for_each_block);
    else
        carry_out_best_merges_advanced(blockmodel, delta_entropy_for_each_block, best_merge_for_each_block, graph);
    blockmodel.distribute(graph);
    blockmodel.initialize_edge_counts(graph);
    MPI_Type_free(&Merge_t);
    return blockmodel;
}

} // namespace block_merge::dist