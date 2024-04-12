#include "entropy.hpp"
#include "fastlgamma.hpp"
#include "spence.hpp"

#include "cmath"

namespace entropy {

double block_merge_delta_mdl(long current_block, long proposal, long num_edges, const Blockmodel &blockmodel,
                             EdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    std::vector<long> old_block_row = blockmodel.blockmatrix()->getrow(current_block); // M_r_t1
    std::vector<long> old_proposal_row = blockmodel.blockmatrix()->getrow(proposal);   // M_s_t1
    std::vector<long> old_block_col = blockmodel.blockmatrix()->getcol(current_block); // M_t2_r
    std::vector<long> old_proposal_col = blockmodel.blockmatrix()->getcol(proposal);   // M_t2_s

    // Exclude current_block, proposal to prevent double counting
    std::vector<long> new_proposal_col = common::exclude_indices(updates.proposal_col, current_block, proposal);
    old_block_col = common::exclude_indices(old_block_col, current_block, proposal);       // M_t2_r
    old_proposal_col = common::exclude_indices(old_proposal_col, current_block, proposal); // M_t2_s
    std::vector<long> new_block_degrees_out = common::exclude_indices(block_degrees.block_degrees_out, current_block,
                                                                     proposal);
    std::vector<long> old_block_degrees_out = common::exclude_indices(blockmodel.degrees_out(),
                                                                     current_block, proposal);

    // Remove 0 indices
    std::vector<long> new_proposal_row_degrees_in = common::index_nonzero(block_degrees.block_degrees_in,
                                                                         updates.proposal_row);
    std::vector<long> new_proposal_row = common::nonzeros(updates.proposal_row);
    std::vector<long> new_proposal_col_degrees_out = common::index_nonzero(new_block_degrees_out, new_proposal_col);
    new_proposal_col = common::nonzeros(new_proposal_col);

    std::vector<long> old_block_row_degrees_in = common::index_nonzero(blockmodel.degrees_in(),
                                                                      old_block_row);
    std::vector<long> old_proposal_row_degrees_in = common::index_nonzero(blockmodel.degrees_in(),
                                                                         old_proposal_row);
    old_block_row = common::nonzeros(old_block_row);
    old_proposal_row = common::nonzeros(old_proposal_row);
    std::vector<long> old_block_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_block_col);
    std::vector<long> old_proposal_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_proposal_col);
    old_block_col = common::nonzeros(old_block_col);
    old_proposal_col = common::nonzeros(old_proposal_col);

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(new_proposal_row, new_proposal_row_degrees_in,
                                                block_degrees.block_degrees_out[proposal], num_edges);
    delta_entropy -= common::delta_entropy_temp(new_proposal_col, new_proposal_col_degrees_out,
                                                block_degrees.block_degrees_in[proposal], num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_row, old_block_row_degrees_in,
                                                blockmodel.degrees_out(current_block), num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_row, old_proposal_row_degrees_in,
                                                blockmodel.degrees_out(proposal), num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_col, old_block_col_degrees_out,
                                                blockmodel.degrees_in(current_block), num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_col, old_proposal_col_degrees_out,
                                                blockmodel.degrees_in(proposal), num_edges);
    return delta_entropy;
}

double block_merge_delta_mdl(long current_block, long proposal, long num_edges, const Blockmodel &blockmodel,
                             SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    const MapVector<long> &old_block_row = matrix->getrow_sparse(current_block); // M_r_t1
    const MapVector<long> &old_proposal_row = matrix->getrow_sparse(proposal);   // M_s_t1
    const MapVector<long> &old_block_col = matrix->getcol_sparse(current_block); // M_t2_r
    const MapVector<long> &old_proposal_col = matrix->getcol_sparse(proposal);   // M_t2_s

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(updates.proposal_row, block_degrees.block_degrees_in,
                                                block_degrees.block_degrees_out[proposal], num_edges);
    delta_entropy -= common::delta_entropy_temp(updates.proposal_col, block_degrees.block_degrees_out,
                                                block_degrees.block_degrees_in[proposal], current_block, proposal,
                                                num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_row, blockmodel.degrees_in(),
                                                blockmodel.degrees_out(current_block), num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_row, blockmodel.degrees_in(),
                                                blockmodel.degrees_out(proposal), num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_col, blockmodel.degrees_out(),
                                                blockmodel.degrees_in(current_block), current_block,
                                                proposal, num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_col, blockmodel.degrees_out(),
                                                blockmodel.degrees_in(proposal), current_block, proposal,
                                                num_edges);
    return delta_entropy;
}

double block_merge_delta_mdl(long current_block, const Blockmodel &blockmodel, const Delta &delta,
                             common::NewBlockDegrees &block_degrees) {
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    double delta_entropy = 0.0;
    long proposed_block = delta.proposed_block();
    for (const std::tuple<long, long, long> &entry: delta.entries()) {
        long row = std::get<0>(entry);
        long col = std::get<1>(entry);
        long change = std::get<2>(entry);
        // delta += + E(old) - E(new)
        delta_entropy += common::cell_entropy(matrix->get(row, col), blockmodel.degrees_in(col),
                                              blockmodel.degrees_out(row));
        if (row == current_block || col == current_block) continue;  // the "new" cell entropy == 0;
        delta_entropy -= common::cell_entropy(matrix->get(row, col) + change, block_degrees.block_degrees_in[col],
                                              block_degrees.block_degrees_out[row]);
    }
    for (const std::pair<const long, long> &entry: blockmodel.blockmatrix()->getrow_sparse(proposed_block)) {
        long row = proposed_block;
        long col = entry.first;
        long value = entry.second;
        if (delta.get(row, col) != 0) continue;
        // Value has not changed
        delta_entropy += common::cell_entropy((double) value, (double) blockmodel.degrees_in(col),
                                              (double) blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy((double) value, (double) block_degrees.block_degrees_in[col],
                                              (double) block_degrees.block_degrees_out[row]);
    }
    for (const std::pair<const long, long> &entry: blockmodel.blockmatrix()->getcol_sparse(proposed_block)) {
        long row = entry.first;
        long col = proposed_block;
        long value = entry.second;
        if (delta.get(row, col) != 0 || row == current_block || row == proposed_block) continue;
        // Value has not changed and we're not double counting
        delta_entropy += common::cell_entropy((double) value, (double) blockmodel.degrees_in(col),
                                              (double) blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy((double) value, (double) block_degrees.block_degrees_in[col],
                                              (double) block_degrees.block_degrees_out[row]);
    }
    return delta_entropy;
}

double block_merge_delta_mdl(long current_block, utils::ProposalAndEdgeCounts proposal, const Blockmodel &blockmodel,
                             const Delta &delta) {
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    double delta_entropy = 0.0;
    long proposed_block = delta.proposed_block();
    auto get_deg_in = [&blockmodel, &proposal, current_block, proposed_block](long index) -> double {
        long value = blockmodel.degrees_in(index);
        if (index == current_block)
            value -= proposal.num_in_neighbor_edges;
        else if (index == proposed_block)
            value += proposal.num_in_neighbor_edges;
        return double(value);
    };
    auto get_deg_out = [&blockmodel, &proposal, current_block, proposed_block](long index) -> double {
        long value = blockmodel.degrees_out(index);
        if (index == current_block)
            value -= proposal.num_out_neighbor_edges;
        else if (index == proposed_block)
            value += proposal.num_out_neighbor_edges;
        return double(value);
    };
    for (const std::tuple<long, long, long> &entry: delta.entries()) {
        long row = std::get<0>(entry);
        long col = std::get<1>(entry);
        auto change = (double) std::get<2>(entry);
        // delta += + E(old) - E(new)
        auto value = (double) matrix->get(row, col);
        delta_entropy += common::cell_entropy(value, (double) blockmodel.degrees_in(col),
                                              (double) blockmodel.degrees_out(row));
        if (row == current_block || col == current_block) continue;  // the "new" cell entropy == 0;
        delta_entropy -= common::cell_entropy(value + change, get_deg_in(col), get_deg_out(row));
    }
    for (const std::pair<long, long> &entry: blockmodel.blockmatrix()->getrow_sparse(proposed_block)) {
        long row = proposed_block;
        long col = entry.first;
        auto value = (double) entry.second;
        if (delta.get(row, col) != 0) continue;
        // Value has not changed
        delta_entropy += common::cell_entropy((double) value, (double) blockmodel.degrees_in(col),
                                              (double) blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy(value, get_deg_in(col), get_deg_out(row));
    }
    for (const std::pair<long, long> &entry: blockmodel.blockmatrix()->getcol_sparse(proposed_block)) {
        long row = entry.first;
        long col = proposed_block;
        auto value = (double) entry.second;
        if (delta.get(row, col) != 0 || row == current_block || row == proposed_block) continue;
        // Value has not changed and we're not double counting
        delta_entropy += common::cell_entropy(value, (double) blockmodel.degrees_in(col),
                                              (double) blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy(value, get_deg_in(col), get_deg_out(row));
    }
    return delta_entropy;
}

double delta_mdl(long current_block, long proposal, const Blockmodel &blockmodel, long num_edges,
                 EdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    std::vector<long> old_block_row = blockmodel.blockmatrix()->getrow(current_block); // M_r_t1
    std::vector<long> old_proposal_row = blockmodel.blockmatrix()->getrow(proposal);   // M_s_t1
    std::vector<long> old_block_col = blockmodel.blockmatrix()->getcol(current_block); // M_t2_r
    std::vector<long> old_proposal_col = blockmodel.blockmatrix()->getcol(proposal);   // M_t2_s

    // Exclude current_block, proposal to prevent double counting
    std::vector<long> new_block_col = common::exclude_indices(updates.block_col, current_block, proposal); // added
    std::vector<long> new_proposal_col = common::exclude_indices(updates.proposal_col, current_block, proposal);
    old_block_col = common::exclude_indices(old_block_col, current_block, proposal);       // M_t2_r
    old_proposal_col = common::exclude_indices(old_proposal_col, current_block, proposal); // M_t2_s
    std::vector<long> new_block_degrees_out = common::exclude_indices(block_degrees.block_degrees_out, current_block,
                                                                     proposal);
    std::vector<long> old_block_degrees_out = common::exclude_indices(blockmodel.degrees_out(), current_block, proposal);

    // Remove 0 indices
    std::vector<long> new_block_row_degrees_in = common::index_nonzero(block_degrees.block_degrees_in,
                                                                      updates.block_row); // added
    std::vector<long> new_proposal_row_degrees_in = common::index_nonzero(block_degrees.block_degrees_in,
                                                                         updates.proposal_row);
    std::vector<long> new_block_row = common::nonzeros(updates.block_row); // added
    std::vector<long> new_proposal_row = common::nonzeros(updates.proposal_row);
    std::vector<long> new_block_col_degrees_out = common::index_nonzero(new_block_degrees_out, new_block_col); // added
    std::vector<long> new_proposal_col_degrees_out = common::index_nonzero(new_block_degrees_out, new_proposal_col);
    new_block_col = common::nonzeros(new_block_col); // added
    new_proposal_col = common::nonzeros(new_proposal_col);

    std::vector<long> old_block_row_degrees_in = common::index_nonzero(blockmodel.degrees_in(), old_block_row);
    std::vector<long> old_proposal_row_degrees_in = common::index_nonzero(blockmodel.degrees_in(), old_proposal_row);
    old_block_row = common::nonzeros(old_block_row);
    old_proposal_row = common::nonzeros(old_proposal_row);
    std::vector<long> old_block_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_block_col);
    std::vector<long> old_proposal_col_degrees_out = common::index_nonzero(old_block_degrees_out, old_proposal_col);
    old_block_col = common::nonzeros(old_block_col);
    old_proposal_col = common::nonzeros(old_proposal_col);

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(new_block_row, new_block_row_degrees_in,
                                                block_degrees.block_degrees_out[current_block], num_edges); // added
    delta_entropy -= common::delta_entropy_temp(new_proposal_row, new_proposal_row_degrees_in,
                                                block_degrees.block_degrees_out[proposal], num_edges);
    delta_entropy -= common::delta_entropy_temp(new_block_col, new_block_col_degrees_out,
                                                block_degrees.block_degrees_in[current_block], num_edges); // added
    delta_entropy -= common::delta_entropy_temp(new_proposal_col, new_proposal_col_degrees_out,
                                                block_degrees.block_degrees_in[proposal], num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_row, old_block_row_degrees_in,
                                                blockmodel.degrees_out(current_block), num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_row, old_proposal_row_degrees_in,
                                                blockmodel.degrees_out(proposal), num_edges);
    delta_entropy += common::delta_entropy_temp(old_block_col, old_block_col_degrees_out,
                                                blockmodel.degrees_in(current_block), num_edges);
    delta_entropy += common::delta_entropy_temp(old_proposal_col, old_proposal_col_degrees_out,
                                                blockmodel.degrees_in(proposal), num_edges);
    if (std::isnan(delta_entropy)) {
        std::cout << "Error: Dense delta entropy is NaN" << std::endl;
        exit(-142321);
    }
    return delta_entropy;
}

double delta_mdl(long current_block, long proposal, const Blockmodel &blockmodel, long num_edges,
                 SparseEdgeCountUpdates &updates, common::NewBlockDegrees &block_degrees) {
    // Blockmodel indexing
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    const MapVector<long> &old_block_row = matrix->getrow_sparseref(current_block); // M_r_t1
    const MapVector<long> &old_proposal_row = matrix->getrow_sparseref(proposal);   // M_s_t1
    const MapVector<long> &old_block_col = matrix->getcol_sparseref(current_block); // M_t2_r
    const MapVector<long> &old_proposal_col = matrix->getcol_sparseref(proposal);   // M_t2_s

    double delta_entropy = 0.0;
    delta_entropy -= common::delta_entropy_temp(updates.block_row, block_degrees.block_degrees_in,
                                                block_degrees.block_degrees_out[current_block], num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy -= common::delta_entropy_temp(updates.proposal_row, block_degrees.block_degrees_in,
                                                block_degrees.block_degrees_out[proposal], num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy -= common::delta_entropy_temp(updates.block_col, block_degrees.block_degrees_out,
                                                block_degrees.block_degrees_in[current_block], current_block, proposal,
                                                num_edges);
    if (std::isnan(delta_entropy)) {
        std::cout << "block_col: ";
        utils::print<long>(updates.block_col);
        std::cout << "_block_degrees_out: ";
        utils::print<long>(block_degrees.block_degrees_out);
        std::cout << "block_degree in: " << block_degrees.block_degrees_in[current_block] << std::endl;
    }
    assert(!std::isnan(delta_entropy));
    delta_entropy -= common::delta_entropy_temp(updates.proposal_col, block_degrees.block_degrees_out,
                                                block_degrees.block_degrees_in[proposal], current_block, proposal,
                                                num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy += common::delta_entropy_temp(old_block_row, blockmodel.degrees_in(),
                                                blockmodel.degrees_out(current_block), num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy += common::delta_entropy_temp(old_proposal_row, blockmodel.degrees_in(),
                                                blockmodel.degrees_out(proposal), num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy += common::delta_entropy_temp(old_block_col, blockmodel.degrees_out(),
                                                blockmodel.degrees_in(current_block), current_block,
                                                proposal, num_edges);
    assert(!std::isnan(delta_entropy));
    delta_entropy += common::delta_entropy_temp(old_proposal_col, blockmodel.degrees_out(),
                                                blockmodel.degrees_in(proposal), current_block, proposal,
                                                num_edges);
    assert(!std::isnan(delta_entropy));
    if (std::isnan(delta_entropy)) {
        std::cerr << "ERROR " << "Error: Sparse delta entropy is NaN" << std::endl;
        exit(-142321);
    }
    return delta_entropy;
}

double delta_mdl(const Blockmodel &blockmodel, const Delta &delta, const utils::ProposalAndEdgeCounts &proposal) {
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    double delta_entropy = 0.0;
    long current_block = delta.current_block();
    long proposed_block = delta.proposed_block();
    auto get_deg_in = [&blockmodel, &proposal, &delta, current_block, proposed_block](long index) -> size_t {
        long value = blockmodel.degrees_in(index);
        if (index == current_block)
            value -= (proposal.num_in_neighbor_edges + delta.self_edge_weight());
        else if (index == proposed_block)
            value += (proposal.num_in_neighbor_edges + delta.self_edge_weight());
        return value;
    };
    auto get_deg_out = [&blockmodel, &proposal, current_block, proposed_block](long index) -> size_t {
        long value = blockmodel.degrees_out(index);
        if (index == current_block)
            value -= proposal.num_out_neighbor_edges;
        else if (index == proposed_block)
            value += proposal.num_out_neighbor_edges;
        return value;
    };
    for (const std::tuple<long, long, long> &entry: delta.entries()) {
        long row = std::get<0>(entry);
        long col = std::get<1>(entry);
        long change = std::get<2>(entry);
        delta_entropy += common::cell_entropy(matrix->get(row, col), blockmodel.degrees_in(col),
                                              blockmodel.degrees_out(row));
        if (std::isnan(delta_entropy) || std::isinf(delta_entropy)) {
            std::cout << delta_entropy << " for row: " << row << " col: " << col << " val: " << matrix->get(row, col) << " delta: " << change << std::endl;
            utils::print<long>(blockmodel.blockmatrix()->getrow_sparse(row));
            utils::print<long>(blockmodel.blockmatrix()->getcol_sparse(col));
            std::cout << "d_out[row]: " << blockmodel.degrees_out(row) << " d_in[col]: " << blockmodel.degrees_in(row) << std::endl;
            throw std::invalid_argument("nan/inf in bm delta for old bm when delta != 0");
        }
        delta_entropy -= common::cell_entropy(matrix->get(row, col) + change, get_deg_in(col),
                                              get_deg_out(row));
        if (std::isnan(delta_entropy) || std::isinf(delta_entropy)) {
            std::cout << delta_entropy << " for row: " << row << " col: " << col << " val: " << matrix->get(row, col) << " delta: " << change;
            std::cout << " current: " << current_block << " proposed: " << proposed_block << std::endl;
            utils::print<long>(blockmodel.blockmatrix()->getrow_sparse(row));
            utils::print<long>(blockmodel.blockmatrix()->getcol_sparse(col));
            std::cout << "d_out[row]: " << blockmodel.degrees_out(row) << " d_in[col]: " << blockmodel.degrees_in(col) << std::endl;
            std::cout << "new d_out[row]: " << get_deg_out(row) << " d_in[col]: " << get_deg_in(col) << std::endl;
            std::cout << "v_out: " << proposal.num_out_neighbor_edges << " v_in: " << proposal.num_in_neighbor_edges << " v_total: " << proposal.num_neighbor_edges << std::endl;
            throw std::invalid_argument("nan/inf in bm delta for new bm when delta != 0");
        }
    }
    // Compute change in entropy for cells with no delta
    for (const auto &entry: blockmodel.blockmatrix()->getrow_sparseref(current_block)) {
        long row = current_block;
        long col = entry.first;
        long value = entry.second;
        if (delta.get(row, col) != 0) continue;
        // Value has not changed
        delta_entropy += common::cell_entropy(value, blockmodel.degrees_in(col),
                                              blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy(value, get_deg_in(col), get_deg_out(row));
        if (std::isnan(delta_entropy) || std::isinf(delta_entropy)) {
            std::cout << delta_entropy << " for row: " << row << " col: " << col << " val: " << value << " delta: 0" << std::endl;
            throw std::invalid_argument("nan/inf in bm delta when delta = 0 and row = current block");
        }
    }
    for (const auto &entry: blockmodel.blockmatrix()->getrow_sparseref(proposed_block)) {
        long row = proposed_block;
        long col = entry.first;
        long value = entry.second;
        if (delta.get(row, col) != 0) continue;
        // Value has not changed
        delta_entropy += common::cell_entropy(value, blockmodel.degrees_in(col),
                                              blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy(value, get_deg_in(col), get_deg_out(row));
        if (std::isnan(delta_entropy) || std::isinf(delta_entropy)) {
            std::cout << delta_entropy << " for row: " << row << " col: " << col << " val: " << value << " delta: 0" << std::endl;
            throw std::invalid_argument("nan/inf in bm delta when delta = 0 and row = proposed block");
        }
    }
    for (const auto &entry: blockmodel.blockmatrix()->getcol_sparseref(current_block)) {
        long row = entry.first;
        long col = current_block;
        long value = entry.second;
        if (delta.get(row, col) != 0 || row == current_block || row == proposed_block) continue;
        // Value has not changed and we're not double counting
        delta_entropy += common::cell_entropy(value, blockmodel.degrees_in(col),
                                              blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy(value, get_deg_in(col), get_deg_out(row));
        if (std::isnan(delta_entropy) || std::isinf(delta_entropy)) {
            std::cout << delta_entropy << " for row: " << row << " col: " << col << " val: " << value << " delta: 0" << std::endl;
            throw std::invalid_argument("nan/inf in bm delta when delta = 0 and col = current block");
        }
    }
    for (const auto &entry: blockmodel.blockmatrix()->getcol_sparseref(proposed_block)) {
        long row = entry.first;
        long col = proposed_block;
        long value = entry.second;
        if (delta.get(row, col) != 0 || row == current_block || row == proposed_block) continue;
        // Value has not changed and we're not double counting
        delta_entropy += common::cell_entropy(value, blockmodel.degrees_in(col),
                                              blockmodel.degrees_out(row));
        delta_entropy -= common::cell_entropy(value, get_deg_in(col), get_deg_out(row));
        if (std::isnan(delta_entropy) || std::isinf(delta_entropy)) {
            std::cout << delta_entropy << " for row: " << row << " col: " << col << " val: " << value << " delta: 0" << std::endl;
            throw std::invalid_argument("nan/inf in bm delta when delta = 0 and col = proposed block");
        }
    }
    return delta_entropy;
}

double hastings_correction(const Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           utils::ProposalAndEdgeCounts &proposal, EdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees) {
    if (proposal.num_neighbor_edges == 0 || args.greedy) {
        return 1.0;
    }
    // Compute block weights
    std::map<long, long> block_counts;
    for (ulong i = 0; i < out_blocks.indices.size(); ++i) {
        long block = out_blocks.indices[i];
        long weight = out_blocks.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    for (ulong i = 0; i < in_blocks.indices.size(); ++i) {
        long block = in_blocks.indices[i];
        long weight = in_blocks.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    // Create Arrays using unique blocks
    size_t num_unique_blocks = block_counts.size();
    std::vector<double> counts(num_unique_blocks, 0);
    std::vector<double> proposal_weights(num_unique_blocks, 0);
    std::vector<double> block_weights(num_unique_blocks, 0);
    std::vector<double> block_degrees(num_unique_blocks, 0);
    std::vector<double> proposal_degrees(num_unique_blocks, 0);
    // Indexing
    std::vector<long> proposal_row = blockmodel.blockmatrix()->getrow(proposal.proposal);
    std::vector<long> proposal_col = blockmodel.blockmatrix()->getcol(proposal.proposal);
    // Fill Arrays
    long index = 0;
    long num_blocks = blockmodel.getNum_blocks();
    const std::vector<long> &current_block_degrees = blockmodel.degrees();
    for (auto const &entry: block_counts) {
        counts[index] = entry.second;
        proposal_weights[index] = proposal_row[entry.first] + proposal_col[entry.first] + 1.0;
        block_degrees[index] = current_block_degrees[entry.first] + num_blocks;
        block_weights[index] = updates.block_row[entry.first] + updates.block_col[entry.first] + 1.0;
        proposal_degrees[index] = new_block_degrees.block_degrees[entry.first] + num_blocks;
        index++;
    }
    // Compute p_forward and p_backward
    auto p_forward = utils::sum<double>(counts * proposal_weights / block_degrees);
    auto p_backward = utils::sum<double>(counts * block_weights / proposal_degrees);
    return p_backward / p_forward;
}

double hastings_correction(const Blockmodel &blockmodel, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                           utils::ProposalAndEdgeCounts &proposal, SparseEdgeCountUpdates &updates,
                           common::NewBlockDegrees &new_block_degrees) {
    if (proposal.num_neighbor_edges == 0 || args.greedy) {
        return 1.0;
    }
    // Compute block weights
    std::map<long, long> block_counts;
    for (ulong i = 0; i < out_blocks.indices.size(); ++i) {
        long block = out_blocks.indices[i];
        long weight = out_blocks.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    for (ulong i = 0; i < in_blocks.indices.size(); ++i) {
        long block = in_blocks.indices[i];
        long weight = in_blocks.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    // Create Arrays using unique blocks
    size_t num_unique_blocks = block_counts.size();
    std::vector<double> counts(num_unique_blocks, 0);
    std::vector<double> proposal_weights(num_unique_blocks, 0);
    std::vector<double> block_weights(num_unique_blocks, 0);
    std::vector<double> block_degrees(num_unique_blocks, 0);
    std::vector<double> proposal_degrees(num_unique_blocks, 0);
    // Indexing
    std::vector<long> proposal_row = blockmodel.blockmatrix()->getrow(proposal.proposal);
    std::vector<long> proposal_col = blockmodel.blockmatrix()->getcol(proposal.proposal);
    // Fill Arrays
    long index = 0;
    long num_blocks = blockmodel.getNum_blocks();
    const std::vector<long> &current_block_degrees = blockmodel.degrees();
    for (auto const &entry: block_counts) {
        counts[index] = entry.second;
        proposal_weights[index] = proposal_row[entry.first] + proposal_col[entry.first] + 1.0;
        block_degrees[index] = current_block_degrees[entry.first] + num_blocks;
        block_weights[index] = updates.block_row[entry.first] + updates.block_col[entry.first] + 1.0;
        proposal_degrees[index] = new_block_degrees.block_degrees[entry.first] + num_blocks;
        index++;
    }
    // Compute p_forward and p_backward
    auto p_forward = utils::sum<double>(counts * proposal_weights / block_degrees);
    auto p_backward = utils::sum<double>(counts * block_weights / proposal_degrees);
    return p_backward / p_forward;
}

double hastings_correction(long vertex, const Graph &graph, const Blockmodel &blockmodel, const Delta &delta,
                           long current_block, const utils::ProposalAndEdgeCounts &proposal) {
    if (proposal.num_neighbor_edges == 0 || args.greedy || args.nonparametric) {  // No correction needed with greedy proposals
        return 1.0;
    }
    // Compute block weights
    MapVector<long> block_counts;
    for (const long neighbor: graph.out_neighbors(vertex)) {
        long neighbor_block = blockmodel.block_assignment(neighbor);
        block_counts[neighbor_block] += 1;
    }
    for (const long neighbor: graph.in_neighbors(vertex)) {
        if (neighbor == vertex) continue;
        long neighbor_block = blockmodel.block_assignment(neighbor);
        block_counts[neighbor_block] += 1;
    }
    // Create Arrays using unique blocks
    size_t num_unique_blocks = block_counts.size();
    std::vector<double> counts(num_unique_blocks, 0);
    std::vector<double> proposal_weights(num_unique_blocks, 0);
    std::vector<double> block_weights(num_unique_blocks, 0);
    std::vector<double> block_degrees(num_unique_blocks, 0);
    std::vector<double> proposal_degrees(num_unique_blocks, 0);
    // Indexing
//    std::vector<long> proposal_row = blockmodel.blockmatrix()->getrow(proposal.proposal);
//    std::vector<long> proposal_col = blockmodel.blockmatrix()->getcol(proposal.proposal);
    const MapVector<long> &proposal_row = blockmodel.blockmatrix()->getrow_sparseref(proposal.proposal);
    const MapVector<long> &proposal_col = blockmodel.blockmatrix()->getcol_sparseref(proposal.proposal);
    // Fill Arrays
    long index = 0;
    long num_blocks = blockmodel.getNum_blocks();
    const std::vector<long> &current_block_degrees = blockmodel.degrees();
    for (auto const &entry: block_counts) {
        counts[index] = entry.second;
        proposal_weights[index] = map_vector::get(proposal_row, entry.first) + map_vector::get(proposal_col, entry.first) + 1.0;
        block_degrees[index] = current_block_degrees[entry.first] + num_blocks;
        block_weights[index] = blockmodel.blockmatrix()->get(current_block, entry.first) +
                               delta.get(current_block, entry.first) +
                               //                get(delta, std::make_pair(current_block, entry.first)) +
                               blockmodel.blockmatrix()->get(entry.first, current_block) +
                               delta.get(entry.first, current_block) + 1.0;
//                get(delta, std::make_pair(entry.first, current_block)) + 1.0;
        long new_block_degree = blockmodel.degrees(entry.first);
        if (entry.first == current_block) {
            long current_block_self_edges = blockmodel.blockmatrix()->get(current_block, current_block)
                                           + delta.get(current_block, current_block);
            long degree_out = blockmodel.degrees_out(current_block) - proposal.num_out_neighbor_edges;
            long degree_in = blockmodel.degrees_in(current_block) - proposal.num_in_neighbor_edges;
            new_block_degree = degree_out + degree_in - current_block_self_edges;
        } else if (entry.first == proposal.proposal) {
            long proposed_block_self_edges = blockmodel.blockmatrix()->get(proposal.proposal, proposal.proposal)
                                            + delta.get(proposal.proposal, proposal.proposal);
            long degree_out = blockmodel.degrees_out(proposal.proposal) + proposal.num_out_neighbor_edges;
            long degree_in = blockmodel.degrees_in(proposal.proposal) + proposal.num_in_neighbor_edges;
            new_block_degree = degree_out + degree_in - proposed_block_self_edges;
        }
//        proposal_degrees[index] = new_block_degrees.block_degrees[entry.first] + num_blocks;
        proposal_degrees[index] = new_block_degree + num_blocks;
        index++;
    }
    // Compute p_forward and p_backward
    auto p_forward = utils::sum<double>(counts * proposal_weights / block_degrees);
    auto p_backward = utils::sum<double>(counts * block_weights / proposal_degrees);
    return p_backward / p_forward;
}

double normalize_mdl_v1(double mdl, const Graph &graph) {
    return mdl / null_mdl_v1(graph);
}

double normalize_mdl_v2(double mdl, long num_vertices, long num_edges) {
    return mdl / null_mdl_v2(num_vertices, num_edges);
}

double null_mdl_v1(const Graph &graph) {
    if (args.nonparametric) {
//        std::cout << "why is this running nonparametric?" << std::endl;
        std::vector<long> assignment = utils::constant<long>(graph.num_vertices(), 0);
        Blockmodel null_model(1, graph, 0.5, assignment);
        return mdl(null_model, graph);
    }
//    std::cout << "running correct version at least..." << std::endl;
    std::cout << graph.num_edges() << std::endl;
    double log_posterior_p = graph.num_edges() * log(1.0 / graph.num_edges());
    double x = 1.0 / graph.num_edges();
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    std::cout << "log posterior = " << log_posterior_p << " blockmodel = " << (graph.num_edges() * h) << std::endl;
    return (graph.num_edges() * h) - log_posterior_p;
}

double null_mdl_v2(long num_vertices, long num_edges) {
    // TODO: not sure how this works in nonparametric version
    double log_posterior_p = num_edges * log(1.0 / num_edges);
    // done calculating log_posterior_probability
    double x = pow(num_vertices, 2) / num_edges;
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
//    std::cout << "log posterior = " << log_posterior_p << " blockmodel = " << (num_edges * h) + (num_vertices * log(num_vertices)) << std::endl;
    return (num_edges * h) + (num_vertices * log(num_vertices)) - log_posterior_p;
}

double mdl(const Blockmodel &blockmodel, const Graph &graph) {
    if (args.nonparametric)
        return nonparametric::mdl(blockmodel, graph);
    double log_posterior_p = blockmodel.log_posterior_probability();
    double x = pow(blockmodel.getNum_blocks(), 2) / graph.num_edges();
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    return (graph.num_edges() * h) + (graph.num_vertices() * log(blockmodel.getNum_blocks())) - log_posterior_p;
}

namespace dist {

double mdl(const TwoHopBlockmodel &blockmodel, long num_vertices, long num_edges) {
    double log_posterior_p = blockmodel.log_posterior_probability();
    double x = pow(blockmodel.getNum_blocks(), 2) / num_edges;
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    return (num_edges * h) + (num_vertices * log(blockmodel.getNum_blocks())) - log_posterior_p;
}

}  // namespace dist

namespace nonparametric {

double get_deg_entropy(const Graph &graph, long vertex) {  // , const simple_degs_t&) {
    long k_in = (long) graph.in_neighbors(vertex).size();
    long k_out = (long) graph.out_neighbors(vertex).size();
    return -fastlgamma(k_in + 1) - fastlgamma(k_out + 1);
}

double sparse_entropy(const Blockmodel &blockmodel, const Graph &graph) {
    double S = 0;

    for (long source = 0; source < blockmodel.getNum_blocks(); ++source) {
        const MapVector<long> &row = blockmodel.blockmatrix()->getrow_sparseref(source);
        for (const std::pair<long, long> &entry : row) {
            long destination = entry.first;
            long weight = entry.second;
            S += eterm_exact(source, destination, weight);
            assert(!std::isinf(S));
            assert(!std::isnan(S));        }
    }

    for (long block = 0; block < blockmodel.getNum_blocks(); ++block) {
        S += vterm_exact(blockmodel.degrees_out(block), blockmodel.degrees_in(block), blockmodel.block_size(block));
        assert(!std::isinf(S));
        assert(!std::isnan(S));    }

//    std::cout << "S before deg_ent?ropy: " << S << std::endl;
    if (!args.degreecorrected) return S;
//    double S2 = 0.0;
    // In distributed case, we would only compute these for vertices we're responsible for. Since it's a simple addition, we can do an allreduce.
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
//        S += get_deg_entropy(graph, vertex);
        double temp = get_deg_entropy(graph, vertex);
        S += temp;
        assert(!std::isinf(S));
        assert(!std::isnan(S));
//        S2 += temp;
    }

//    std::cout << "deg_entropy: " << S2 << std::endl;

    return S;
}

double get_partition_dl(long N, const Blockmodel &blockmodel) { // _N = number of vertices, _actual_B = nonzero blocks, _total = vector of block sizes
    double S = 0;
    S += fastlbinom(N - 1, blockmodel.num_nonempty_blocks() - 1);
    S += fastlgamma(N + 1);
    for (const long &block_size : blockmodel.block_sizes())
        S -= fastlgamma(block_size + 1);
    S += fastlog(N);
//    std::cout << "Partition dl: " << S << std::endl;
    assert(!std::isinf(S));
    assert(!std::isnan(S));
    return S;
}

/// No idea what this function does. See int_part.cc in https://git.skewed.de/count0/graph-tool
double get_v(double u, double epsilon) {
    double v = u;
    double delta = 1;
    while (delta > epsilon) {
        double n_v = u * sqrt(spence(exp(-v)));
        delta = abs(n_v - v);
        v = n_v;
    }
    return v;
}

double log_q_approx_small(size_t n, size_t k) {
    return fastlbinom(n - 1, k - 1) - fastlgamma(k + 1);
}

/// Computes the number of restricted of integer n into at most m parts. This is part of teh prior for the
/// degree-corrected SBM.
/// TO-DO: the current function contains only the approximation of log_q. If it becomes a bottleneck, you'll want to
/// compute a cache of log_q(n, m) for ~20k n and maybe a few hundred m? I feel like for larger graphs, the cache
/// will be a waste of time.
/// See int_part.cc in https://git.skewed.de/count0/graph-tool
double log_q(size_t n, size_t k) {
    if (n <= 0 || k < 1) return 0;
    if (k > n) k = n;
    if (k < pow(n, 1/4.))
        return log_q_approx_small(n, k);
    double u = k / sqrt(n);
    double v = get_v(u);
    double lf = log(v) - log1p(- exp(-v) * (1 + u * u/2)) / 2 - log(2) * 3 / 2.
                - log(u) - log(M_PI);
    double g = 2 * v / u - u * log1p(-exp(-v));
    return lf - log(n) + sqrt(n) * g;
}

double get_deg_dl_dist(const Blockmodel &blockmodel) { // Rs&& rs, Ks&& ks) {  // RS: range from 0 to B, KS is an empty array of pairs?
    if (!args.degreecorrected) return 0.0;

    double S = 0;

    for (int block = 0; block < blockmodel.getNum_blocks(); ++block) {
        S += log_q(blockmodel.degrees_out(block), blockmodel.block_size(block));
        S += log_q(blockmodel.degrees_in(block), blockmodel.block_size(block));
        size_t total = 0;
        if (!args.undirected) {
            for (const std::pair<long, long> &entry : blockmodel.in_degree_histogram(block)) {
                S -= fastlgamma(entry.second + 1);
                assert(!std::isinf(S));
                assert(!std::isnan(S));
            }
        }
        for (const std::pair<long, long> &entry : blockmodel.out_degree_histogram(block)) {
            S -= fastlgamma(entry.second + 1);
            assert(!std::isinf(S));
            assert(!std::isnan(S));
            total += entry.second;
        }

        if (args.undirected) {
            S += fastlgamma(total + 1);
        } else {
            S += 2 * fastlgamma(total + 1);
        }
        assert(!std::isinf(S));
        assert(!std::isnan(S));
    }
//    std::cout << "degree_dl: " << S << std::endl;
    return S;
}

double get_edges_dl(size_t B, size_t E) {
    size_t NB = !args.undirected ? B * B : (B * (B + 1)) / 2;
    double E_dl = fastlbinom(NB + E - 1, E);
//    std::cout << "edges_dl: " << E_dl << std::endl;
    return E_dl;
}

double mdl(const Blockmodel &blockmodel, const Graph &graph) {
    double S = 0, S_dl = 0;

    S = sparse_entropy(blockmodel, graph);
    assert(!std::isinf(S));
    assert(!std::isnan(S));
//    std::cout << "sparse E: " << S << std::endl;

    S_dl += get_partition_dl(graph.num_vertices(), blockmodel);
    assert(!std::isinf(S_dl));
    assert(!std::isnan(S_dl));
//    std::cout << "partition_dl: " << S_dl << std::endl;

    S_dl += get_deg_dl_dist(blockmodel);  // (ea.degree_dl_kind);
    assert(!std::isinf(S_dl));
    assert(!std::isnan(S_dl));
//    std::cout << "after deg_dl: " << S_dl << std::endl;

//    std::cout << "NB: " << blockmodel.num_nonempty_blocks() << " E: " << graph.num_edges() << std::endl;
//    S_dl += get_edges_dl(blockmodel.num_nonempty_blocks(), graph.num_edges());
    double E_dl = get_edges_dl(blockmodel.num_nonempty_blocks(), graph.num_edges());
//    std::cout << "edges_dl: " << E_dl << std::endl;
    S_dl += E_dl;
    assert(!std::isinf(S_dl));
    assert(!std::isnan(S_dl));
//    std::cout << "after edge_dl: " << S_dl << std::endl;
//    utils::print(blockmodel.block_sizes());
    return S + S_dl * BETA_DL;
}

// obtain the entropy difference given a set of entries in the e_rs matrix
//template <bool exact, class MEntries, class Eprop, class EMat, class BGraph>
//[[gnu::always_inline]] [[gnu::flatten]] [[gnu::hot]] inline
double entries_dS(const Blockmodel &blockmodel, const Delta &delta) {  // MEntries& m_entries, Eprop& mrs, EMat& emat, BGraph& bg) {
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    double dS = 0;
    for (const std::tuple<long, long, long> &entry : delta.entries()) {
        long row = std::get<0>(entry);
        long col = std::get<1>(entry);
        auto change = (long) std::get<2>(entry);
        // delta += + E(old) - E(new)
        auto value = (long) matrix->get(row, col);
        dS += eterm_exact(row, col, value + change) - eterm_exact(row, col, value);
        assert(!std::isinf(dS));
        assert(!std::isnan(dS));
    }
//    entries_op(m_entries, emat,
//               [&](auto r, auto s, auto& me, auto d)
//               {
//                   size_t ers = 0;
//                   if (me != emat.get_null_edge())
//                       ers = mrs[me];
//                   assert(int(ers) + d >= 0);
//                   if constexpr (exact)
//                       dS += eterm_exact(r, s, ers + d, bg) - eterm_exact(r, s, ers, bg);
//                   else
//                       dS += eterm(r, s, ers + d, bg) - eterm(r, s, ers, bg);
//               });
    return dS;
}

// compute the entropy difference of a virtual move of vertex from block r
// to nr
//template <bool exact, class MEntries>
double virtual_move_sparse(const Blockmodel &blockmodel, const Delta &delta, long weight,
                           const utils::ProposalAndEdgeCounts &proposal) {  // size_t v, size_t r, size_t nr, MEntries& m_entries) {
    if (delta.current_block() == delta.proposed_block()) return 0.;

    double dS = entries_dS(blockmodel, delta);  // <exact>(m_entries, _mrs, _emat, _bg);

    long kin = proposal.num_in_neighbor_edges;
    long kout = proposal.num_out_neighbor_edges;
//    auto [kin, kout] = get_deg(v, _eweight, _degs, _g);

//    int dwr = _vweight[v];
//    int dwnr = dwr;
//    int dwr = 1, dwnr = 1;

//    if (r == null_group && dwnr == 0)
//        dwnr = 1;

    auto vt = [&](auto out_degree, auto in_degree, auto w) { // , auto nr) {
        assert(out_degree >= 0 && in_degree >=0);
//        if constexpr (exact)
        return vterm_exact(out_degree, in_degree, w);  // , nr, _deg_corr, _bg);
//        else
//            return vterm(mrp, mrm, nr, _deg_corr, _bg);
    };

//    if (r != null_group)
//    {
//    auto mrp_r = _mrp[r];
//    auto mrm_r = _mrm[r];
//    auto wr_r = _wr[r];
    dS += vt(blockmodel.degrees_out(delta.current_block()) - kout, blockmodel.degrees_in(delta.current_block()) - kin, blockmodel.block_size(delta.current_block()) - weight);  // , wr_r - dwr);
    dS -= vt(blockmodel.degrees_out(delta.current_block()), blockmodel.degrees_in(delta.current_block()), blockmodel.block_size(delta.current_block()));  //        , mrm_r      , wr_r      );
    assert(!std::isinf(dS));
    assert(!std::isnan(dS));
//    }

//    if (nr != null_group)
//    {
//        auto mrp_nr = _mrp[nr];
//        auto mrm_nr = _mrm[nr];
//        auto wr_nr = _wr[nr];
    dS += vt(blockmodel.degrees_out(delta.proposed_block()) + kout, blockmodel.degrees_in(delta.proposed_block()) + kin, blockmodel.block_size(delta.proposed_block()) + weight);  // , wr_nr + dwnr);
    dS -= vt(blockmodel.degrees_out(delta.proposed_block()), blockmodel.degrees_in(delta.proposed_block()), blockmodel.block_size(delta.proposed_block()));  //        , mrm_nr      , wr_nr       );
    assert(!std::isinf(dS));
    assert(!std::isnan(dS));
//    }

//    std::cout << "delta sparse entropy: " << dS << std::endl;
    return dS;
}

double get_delta_partition_dl(long num_vertices, const Blockmodel &blockmodel, const Delta &delta, long weight) {  // size_t v, size_t r, size_t nr, const entropy_args_t& ea) {
    if (delta.current_block() == delta.proposed_block()) return 0.;

    double dS = 0;

//    auto& f = _bfield[v];
//    if (!f.empty())
//    {
//        if (nr != null_group)
//            dS -= (nr < f.size()) ? f[nr] : f.back();
//        if (r != null_group)
//            dS += (r < f.size()) ? f[r] : f.back();
//    }

//    if (r == nr)
//        return 0;

//    if (r != null_group)
//        r = get_r(r);

//    if (nr != null_group)
//        nr = get_r(nr);

//    int n = 1;  // vweight[v]; for block_merge, change this to size of the blockmodel
//    if (n == 0) {
//        if (r == null_group)
//            n = 1;
//        else
//            return 0;
//    }

    double S_b = 0;
    double S_a = 0;

//    if (r != null_group)
//    {
//    std::cout << "size of current: " << blockmodel.block_size(delta.current_block())
    S_b += -fastlgamma(blockmodel.block_size(delta.current_block()) + 1);  // _total[r] + 1);
    S_a += -fastlgamma(blockmodel.block_size(delta.current_block()) - weight + 1);  // _total[r] - n + 1);

//    std::cout << "point A: S_b = " << S_b << " S_a = " << S_a << std::endl;
//    }

//    if (nr != null_group)
//    {
    S_b += -fastlgamma(blockmodel.block_size(delta.proposed_block()) + 1);  // _total[nr] + 1);
    S_a += -fastlgamma(blockmodel.block_size(delta.proposed_block()) + weight + 1);  // _total[nr] + n + 1);

//    std::cout << "point B: S_b = " << S_b << " S_a = " << S_a << std::endl;

//    }

//    int dN = 0;
//    if (r == null_group)
//        dN += n;
//    if (nr == null_group)
//        dN -= n;
//
//    S_b += lgamma_fast(_N + 1);
//    S_a += lgamma_fast(_N + dN + 1);

    int dB = 0;
    if (blockmodel.block_size(delta.current_block()) == weight)
        dB--;
    if (blockmodel.block_size(delta.proposed_block()) == 0)
        dB++;
//    if (r != null_group && _total[r] == n)
//        dB--;
//    if (nr != null_group && _total[nr] == 0)
//        dB++;

    if (dB != 0) {
        S_b += fastlbinom(num_vertices - 1, blockmodel.num_nonempty_blocks() - 1);
        S_a += fastlbinom(num_vertices - 1, blockmodel.num_nonempty_blocks() + dB - 1);
    }

//    std::cout << "point C: S_b = " << S_b << " S_a = " << S_a << std::endl;

//    if ((dN != 0 || dB != 0)) {
//        S_b += lbinom_fast(_N - 1, _actual_B - 1);
//        S_a += lbinom_fast(_N - 1 + dN, _actual_B + dB - 1);
//    }

//    if (dN != 0)
//    {
//        S_b += safelog_fast(_N);
//        S_a += safelog_fast(_N + dN);
//    }

    dS += S_a - S_b;
    assert(!std::isinf(dS));
    assert(!std::isnan(dS));

//    if (ea.partition_dl)
//    {
//        auto& ps = get_partition_stats(v);
//        dS += ps.get_delta_partition_dl(v, r, nr, _vweight);
//    }

//    if (_coupled_state != nullptr)
//    {
//        bool r_vacate = (r != null_group && _wr[r] == _vweight[v]);
//        bool nr_occupy = (nr != null_group && _wr[nr] == 0);
//
//        auto& bh = _coupled_state->get_b();
//        if (r_vacate && nr_occupy)
//        {
//            dS += _coupled_state->get_delta_partition_dl(r, bh[r], bh[nr],
//                                                         _coupled_entropy_args);
//        }
//        else
//        {
//            if (r_vacate)
//                dS += _coupled_state->get_delta_partition_dl(r, bh[r], null_group,
//                                                             _coupled_entropy_args);
//            if (nr_occupy)
//                dS += _coupled_state->get_delta_partition_dl(nr, null_group, bh[nr],
//                                                             _coupled_entropy_args);
//        }
//    }
//    std::cout << "delta partition dl: " << dS << std::endl;
    return dS;
}

//template <class DegOP>
//double get_delta_deg_dl_dist_change(size_t r, DegOP&& dop, int diff) {
double get_delta_deg_dl_dist_change(const Blockmodel &blockmodel, long block, long vkin, long vkout, long vweight,
                                    int diff) {
//    if (!args.degreecorrected) return 0.0;
    // vweight may be unnecessary. At least for the DOp portion
    auto total_r = blockmodel.block_size(block);  // _total[r];
    auto ep_r = blockmodel.degrees_out(block);  // _ep[r];
    auto em_r = blockmodel.degrees_in(block);  // _em[r];

    auto get_Se = [&](int delta, int kin, int kout) {
        double S = 0;
        assert(total_r + delta >= 0);
        assert(em_r + kin >= 0);
        assert(ep_r + kout >= 0);
        S += log_q(em_r + kin, total_r + delta);
        S += log_q(ep_r + kout, total_r + delta);
        return S;
    };

    auto get_Sr = [&](int delta) {
        assert(total_r + delta + 1 >= 0);
        if (args.undirected)
            return fastlgamma(total_r + delta + 1);
        else
            return 2 * fastlgamma(total_r + delta + 1);
    };

    auto get_Sk = [&](std::pair<long, long>& deg, int delta) {
        double S = 0;
        int nd = 0;
        if (!args.undirected) {
//            if (_hist_in[block] != nullptr) {
//                auto& h = *_hist_in[block];
//            if (blockmodel.in_degree_histogram(block) != nullptr) {
            const MapVector<long> &histogram = blockmodel.in_degree_histogram(block);
            auto iter = histogram.find(std::get<0>(deg));
            if (iter != histogram.end())
                nd = iter->second;
//            }
            S -= fastlgamma(nd + delta + 1);
        }

        nd = 0;
//        if (_hist_out[block] != nullptr) {
//        auto& h = *_hist_out[block];
        const MapVector<long> &histogram = blockmodel.out_degree_histogram(block);
        auto iter = histogram.find(std::get<1>(deg));
        if (iter != histogram.end())
            nd = iter->second;
//        }

        return S - fastlgamma(nd + delta + 1);
    };

    double S_b = 0, S_a = 0;
    int tkin = 0, tkout = 0, n = 0;
//    dop([&](size_t kin, size_t kout, int nk)
//        {
    tkin += vkin;  //  * vweight;
    tkout += vkout;  //  * vweight;
    n += vweight;

    std::pair<long, long> deg = std::make_pair(vkin, vkout);
    S_b += get_Sk(deg,         0);
    S_a += get_Sk(deg, diff * vweight);
//        });

    S_b += get_Se(       0,           0,            0);
    S_a += get_Se(diff * n, diff * tkin, diff * tkout);

    S_b += get_Sr(       0);
    S_a += get_Sr(diff * n);

    return S_a - S_b;
}


//template <class Graph, class VProp, class EProp, class Degs>
double get_delta_deg_dl(long vertex, const Blockmodel &blockmodel, const Delta &delta, const Graph &graph) {  // size_t r, size_t nr, VProp& vweight,
//                        EProp& eweight, Degs& degs, Graph& g, int kind) {
//    if (r == nr || vweight[v] == 0)
//        return 0;
    if (!args.degreecorrected) return 0.00;
    if (delta.current_block() == delta.proposed_block()) return 0.;  // for block_merge, it's this || size(block) == 0
//    if (r != null_group)
//        r = get_r(r);
//    if (nr != null_group)
//        nr = get_r(nr);

//    auto dop = [&](auto&& f) {
//        long kin = graph.in_neighbors(vertex).size();
//        long kout = graph.out_neighbors(vertex).size();
////        auto [kin, kout] = get_deg(v, eweight, degs, g);
//        f(kin, kout, 1);  // for block merge, it's size(block) instead of 1 | vweight[v]);
//    };

    long vkin = graph.in_neighbors(vertex).size();
    long vkout = graph.out_neighbors(vertex).size();
    double dS = 0;
//    if (r != null_group)
//    dS += get_delta_deg_dl_dist_change(blockmodel, delta.current_block(),  dop, -1);
    dS += get_delta_deg_dl_dist_change(blockmodel, delta.current_block(),  vkin, vkout, 1, -1);
//    if (nr != null_group)
//    dS += get_delta_deg_dl_dist_change(blockmodel, delta.proposed_block(), dop, +1);
    dS += get_delta_deg_dl_dist_change(blockmodel, delta.proposed_block(), vkin, vkout, 1, +1);
    assert(!std::isinf(dS));
    assert(!std::isnan(dS));

//    }
//    std::cout << "delta degree dl: " << dS << std::endl;
    return dS;
}

//template <class VProp, class Graph>
double get_delta_edges_dl(const Blockmodel &blockmodel, const Delta &delta, long weight, long num_edges) {
    if (delta.current_block() == delta.proposed_block())
        return 0;

//    if (r != null_group)
//        r = get_r(r);
//    if (nr != null_group)
//        nr = get_r(nr);

    double S_b = 0, S_a = 0;

//    int n = weight;  // vweight[v];

//    if (n == 0)
//    {
//        if (r == null_group)
//            n = 1;
//        else
//            return 0;
//    }

    int dB = 0;
//    if (r != null_group && _total[r] == n)
    if (blockmodel.block_size(delta.current_block()) == weight)
        dB--;
//    if (nr != null_group && _total[nr] == 0)
    if (blockmodel.block_size(delta.proposed_block()) == 0)
        dB++;

    if (dB != 0) {
        S_b += get_edges_dl(blockmodel.num_nonempty_blocks(), num_edges);
        S_a += get_edges_dl(blockmodel.num_nonempty_blocks() + dB, num_edges);
    }

    double dS = S_a - S_b;
    assert(!std::isinf(dS));
    assert(!std::isnan(dS));
//    std::cout << "delta edges dl: " << dS << std::endl;
    return dS;
}

double delta_mdl(const Blockmodel &blockmodel, const Graph &graph, long vertex, const Delta &delta,
                 const utils::ProposalAndEdgeCounts &proposal) {
//    std::cout << blockmodel.block_assignment(vertex) << " != " << delta.current_block() << std::endl;
    assert(blockmodel.block_assignment(vertex) == delta.current_block());

//    get_move_entries(v, r, nr, m_entries, [](auto) constexpr { return false; });

    if (delta.current_block() == delta.proposed_block()) return 0;
//    if (r == nr || _vweight[v] == 0)
//        return 0;

    double dS = 0;
    dS = virtual_move_sparse(blockmodel, delta, 1, proposal);  // <true>(v, r, nr, m_entries);

    double dS_dl = 0;

    dS_dl += get_delta_partition_dl(graph.num_vertices(), blockmodel, delta, 1);  // v, r, nr, ea);
    assert(!std::isinf(dS_dl));
    assert(!std::isnan(dS_dl));

//    if (ea.degree_dl || ea.edges_dl) {
//    auto& ps = get_partition_stats(v);
//    if (_deg_corr && ea.degree_dl)
    dS_dl += get_delta_deg_dl(vertex, blockmodel, delta, graph);  // v, r, nr, _vweight, _eweight, _degs, _g, ea.degree_dl_kind);
//    if (ea.edges_dl)
//    {
//    size_t actual_B = 0;
//    for (auto& ps : _partition_stats)
//        actual_B += ps.get_actual_B();
    dS_dl += get_delta_edges_dl(blockmodel, delta, 1, graph.num_edges());  // v, r, nr, _vweight, actual_B, _g);

    return dS + BETA_DL * dS_dl;
}

double get_delta_deg_dl(const Blockmodel &blockmodel, const Delta &delta) {
    if (!args.degreecorrected) return 0.0;

    if (delta.current_block() == delta.proposed_block()) return 0.;  // for block_merge, it's this || size(block) == 0

    long vkin = blockmodel.degrees_in(delta.current_block());  // graph.in_neighbors(vertex).size();
    long vkout = blockmodel.degrees_out(delta.current_block());  // graph.out_neighbors(vertex).size();
    long weight = blockmodel.block_size(delta.current_block());
    double dS = 0;
//    dS += get_delta_deg_dl_dist_change(blockmodel, delta.current_block(),  dop, -1);
    dS += get_delta_deg_dl_dist_change(blockmodel, delta.current_block(),  vkin, vkout, weight, -1);
//    dS += get_delta_deg_dl_dist_change(blockmodel, delta.proposed_block(), dop, +1);
    dS += get_delta_deg_dl_dist_change(blockmodel, delta.proposed_block(), vkin, vkout, weight, +1);

//    }
    return dS;
}

double block_merge_delta_mdl(const Blockmodel &blockmodel, const utils::ProposalAndEdgeCounts &proposal,
                             const Graph &graph, const Delta &delta) {
//    get_move_entries(v, r, nr, m_entries, [](auto) constexpr { return false; });

    if (delta.current_block() == delta.proposed_block()) return 0;

    if (blockmodel.block_size(delta.current_block()) == 0 || blockmodel.block_size(delta.proposed_block()) == 0)
        return 0;
//    if (r == nr || _vweight[v] == 0)
//        return 0;

    double dS = 0;

    dS = virtual_move_sparse(blockmodel, delta, blockmodel.block_size(delta.current_block()), proposal);  // <true>(v, r, nr, m_entries);

    double dS_dl = 0;

    dS_dl += get_delta_partition_dl(graph.num_vertices(), blockmodel, delta,
                                    blockmodel.block_size(delta.current_block()));  // v, r, nr, ea);

    dS_dl += get_delta_deg_dl(blockmodel, delta);  // v, r, nr, _vweight, _eweight, _degs, _g, ea.degree_dl_kind);

    dS_dl += get_delta_edges_dl(blockmodel, delta, blockmodel.block_size(delta.current_block()), graph.num_edges());  // v, r, nr, _vweight, actual_B, _g);

    return dS + BETA_DL * dS_dl;
}

}  // namespace nonparametric

}  // namespace entropy
