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
#include "entropy.hpp"

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
    if (proposal.num_neighbor_edges == 0 || args.greedy) {  // No correction needed with greedy proposals
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

double normalize_mdl_v1(double mdl, long num_edges) {
    return mdl / null_mdl_v1(num_edges);
}

double normalize_mdl_v2(double mdl, long num_vertices, long num_edges) {
    return mdl / null_mdl_v2(num_vertices, num_edges);
}

double null_mdl_v1(long num_edges) {
    double log_posterior_p = num_edges * log(1.0 / num_edges);
    double x = 1.0 / num_edges;
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
//    std::cout << "log posterior = " << log_posterior_p << " blockmodel = " << (num_edges * h) << std::endl;
    return (num_edges * h) - log_posterior_p;
}

double null_mdl_v2(long num_vertices, long num_edges) {
    double log_posterior_p = num_edges * log(1.0 / num_edges);
    // done calculating log_posterior_probability
    double x = pow(num_vertices, 2) / num_edges;
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
//    std::cout << "log posterior = " << log_posterior_p << " blockmodel = " << (num_edges * h) + (num_vertices * log(num_vertices)) << std::endl;
    return (num_edges * h) + (num_vertices * log(num_vertices)) - log_posterior_p;
}

double mdl(const Blockmodel &blockmodel, long num_vertices, long num_edges) {
    double log_posterior_p = blockmodel.log_posterior_probability();
    double x = pow(blockmodel.getNum_blocks(), 2) / num_edges;
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    return (num_edges * h) + (num_vertices * log(blockmodel.getNum_blocks())) - log_posterior_p;
}

namespace dist {

double mdl(const TwoHopBlockmodel &blockmodel, long num_vertices, long num_edges) {
    double log_posterior_p = blockmodel.log_posterior_probability();
    double x = pow(blockmodel.getNum_blocks(), 2) / num_edges;
    double h = ((1 + x) * log(1 + x)) - (x * log(x));
    return (num_edges * h) + (num_vertices * log(blockmodel.getNum_blocks())) - log_posterior_p;
}

}  // namespace dist

}  // namespace entropy
