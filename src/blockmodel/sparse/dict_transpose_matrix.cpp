#include "dict_transpose_matrix.hpp"

void DictTransposeMatrix::add(long row, long col, long val) {
    // TODO: bound checking includes branching, may get rid of it for performance
    check_row_bounds(row);
    check_col_bounds(col);
    this->matrix[row][col] += val;
//    this->matrix_transpose[col][row] += val;
}

void DictTransposeMatrix::add_transpose(long row, long col, long val) {
    // TODO: bound checking includes branching, may get rid of it for performance
    check_row_bounds(row);
    check_col_bounds(col);
    this->matrix_transpose[col][row] += val;
}

// void DictTransposeMatrix::check_row_bounds(long row) {
//     if (row < 0 || row >= this->nrows) {
//         throw IndexOutOfBoundsException(row, this->nrows);
//     }
// }

// void DictTransposeMatrix::check_col_bounds(long col) {
//     if (col < 0 || col >= this->ncols) {
//         throw IndexOutOfBoundsException(col, this->ncols);
//     }
// }

// void DictTransposeMatrix::check_row_bounds(long row) const {
//     if (row < 0 || row >= this->nrows) {
//         throw IndexOutOfBoundsException(row, this->nrows);
//     }
// }

// void DictTransposeMatrix::check_col_bounds(long col) const {
//     if (col < 0 || col >= this->ncols) {
//         throw IndexOutOfBoundsException(col, this->ncols);
//     }
// }

void DictTransposeMatrix::clearcol(long col) {
    this->matrix_transpose[col].clear();
    for (MapVector<long> &row : this->matrix) {
        row.erase(col);
    }
}

void DictTransposeMatrix::clearrow(long row) {
    this->matrix[row].clear();
    for (MapVector<long> &col : this->matrix_transpose) {
        col.erase(row);
    }
}

ISparseMatrix* DictTransposeMatrix::copy() const {
    // std::vector<std::unordered_map<long, long>> dict_matrix(this->nrows, std::unordered_map<long, long>());
    DictTransposeMatrix dict_matrix(this->nrows, this->ncols);
    for (long i = 0; i < this->nrows; ++i) {
        const MapVector<long> row = this->matrix[i];
        dict_matrix.matrix[i] = row;  // TODO: double-check that this is a copy constructor
    }
    for (long i = 0; i < this->ncols; ++i) {
        const MapVector<long> col = this->matrix_transpose[i];
        dict_matrix.matrix_transpose[i] = col;
    }
    return new DictTransposeMatrix(dict_matrix);
}

long DictTransposeMatrix::distinct_edges(long block) const {
    long result = (long) this->matrix[block].size();
    result += (long) this->matrix_transpose[block].size();
    if (this->matrix[block].contains(block)) {  // If block has a self edge, only count it once.
        result -= 1;
    }
    return result;
}

std::vector<std::tuple<long, long, long>> DictTransposeMatrix::entries() const {
    throw std::logic_error("entries() is not implemented for DictTransposeMatrix!");
}

long DictTransposeMatrix::get(long row, long col) const {
    check_row_bounds(row);
    check_col_bounds(col);
    const MapVector<long> &row_vector = this->matrix[row];
    auto it = row_vector.find(col);
    if (it == row_vector.end())
        return 0;
    return it->second;
    // return this->matrix[row][col];
}

std::vector<long> DictTransposeMatrix::getcol(long col) const {
    check_col_bounds(col);
    std::vector<long> col_values(this->nrows, 0);
    const MapVector<long> &matrix_col = this->matrix_transpose[col];
    for (const std::pair<long, long> element : matrix_col) {
        col_values[element.first] = element.second;
    }
    return col_values;
}

MapVector<long> DictTransposeMatrix::getcol_sparse(long col) const {
    check_col_bounds(col);
    return this->matrix_transpose[col];
}

const MapVector<long>& DictTransposeMatrix::getcol_sparseref(long col) const {
    check_row_bounds(col);
    return this->matrix_transpose[col];
}

// const MapVector<long>& DictTransposeMatrix::getcol_sparse(long col) const {
//     check_col_bounds(col);
//     return this->matrix_transpose[col];
// }

void DictTransposeMatrix::getcol_sparse(long col, MapVector<long> &col_vector) const {
    check_col_bounds(col);
    col_vector = this->matrix_transpose[col];
}

std::vector<long> DictTransposeMatrix::getrow(long row) const {
    check_row_bounds(row);
    std::vector<long> row_values = utils::constant<long>(this->ncols, 0);
    // long row_values [this->ncols];
    // NOTE: could save some time by pulling this->matrix[row] out, and then iterating over it using references
    // but, this could be not thread safe
    // for (long row_index = 0; row_index < this->nrows; ++row_index) {
    const MapVector<long> &matrix_row = this->matrix[row];
    for (const std::pair<long, long> element : matrix_row) {
        row_values[element.first] = element.second;
    }
    return row_values;
}

MapVector<long> DictTransposeMatrix::getrow_sparse(long row) const {
    check_row_bounds(row);
    return this->matrix[row];
}

 const MapVector<long>& DictTransposeMatrix::getrow_sparseref(long row) const {
     check_row_bounds(row);
     return this->matrix[row];
 }

void DictTransposeMatrix::getrow_sparse(long row, MapVector<long> &row_vector) const {
    check_row_bounds(row);
    row_vector = this->matrix_transpose[row];
}

EdgeWeights DictTransposeMatrix::incoming_edges(long block) const {
    check_col_bounds(block);
    std::vector<long> indices;
    std::vector<long> values;
    const MapVector<long> &block_col = this->matrix_transpose[block];
    for (const std::pair<long, long> &element : block_col) {
        indices.push_back(element.first);
        values.push_back(element.second);
    }
    return EdgeWeights {indices, values};
}

std::set<long> DictTransposeMatrix::neighbors(long block) const {
    std::set<long> result;
    for (const std::pair<long, long> &entry : this->matrix[block]) {
        result.insert(entry.first);
    }
    for (const std::pair<long, long> &entry : this->matrix_transpose[block]) {
        result.insert(entry.first);
    }
    return result;
}

MapVector<long> DictTransposeMatrix::neighbors_weights(long block) const {
    MapVector<long> result(this->matrix[block].size() + this->matrix_transpose[block].size());
    for (const std::pair<long, long> &entry : this->matrix[block]) {
        result[entry.first] += entry.second;
//        result.insert(entry.first);
    }
    for (const std::pair<long, long> &entry : this->matrix_transpose[block]) {
        if (entry.first == block) continue;
        result[entry.first] += entry.second;
//        result.insert(entry.first);
    }
    return result;
}

Indices DictTransposeMatrix::nonzero() const {
    std::vector<long> row_vector;
    std::vector<long> col_vector;
    for (long row = 0; row < nrows; ++row) {
        const MapVector<long> matrix_row = this->matrix[row];
        for (const std::pair<const long, long> &element : matrix_row) {
            row_vector.push_back(row);
            col_vector.push_back(element.first);
        }
    }
    return Indices{row_vector, col_vector};
}

void DictTransposeMatrix::setcol(long col, const MapVector<long> &vector) {
    this->matrix_transpose[col] = MapVector<long>(vector);
    for (long row = 0; row < (long) this->matrix.size(); ++row) {
        MapVector<long>::const_iterator value = vector.find(row);
        if (value == vector.end())  // value is not in vector
            this->matrix[row].erase(col);
        else
            this->matrix[row][col] = value->second;
    }
}

void DictTransposeMatrix::setrow(long row, const MapVector<long> &vector) {
    this->matrix[row] = MapVector<long>(vector);
    for (long col = 0; col < (long) this->matrix.size(); ++col) {
        MapVector<long>::const_iterator value = vector.find(col);
        if (value == vector.end())  // value is not in vector
            this->matrix_transpose[col].erase(row);
        else
            this->matrix_transpose[col][row] = value->second;
    }
}

void DictTransposeMatrix::sub(long row, long col, long val) {
    check_row_bounds(row);
    check_col_bounds(col);
    // TODO: debug mode - if matrix[row][col] doesn't exist, throw exception
    this->matrix[row][col] -= val;
    this->matrix_transpose[col][row] -= val;
    if (this->matrix[row][col] == 0) {
        this->matrix[row].erase(col);
        this->matrix_transpose[col].erase(row);
    }
}

long DictTransposeMatrix::edges() const {
    long total = 0;
    for (long row = 0; row < nrows; ++row) {
        const MapVector<long> &matrix_row = this->matrix[row];
        for (const std::pair<long, long> &element : matrix_row) {
            total += element.second;
        }
    }
    return total;
}

void DictTransposeMatrix::print() const {
    std::cout << "Matrix: " << std::endl;
    for (long row = 0; row < this->nrows; ++row) {
        for (long col = 0; col < this->ncols; ++col) {
            std::cout << map_vector::get(this->matrix[row], col) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Transpose: " << std::endl;
    for (long col = 0; col < this->ncols; ++col) {
        for (long row = 0; row < this->nrows; ++row) {
            std::cout << map_vector::get(this->matrix_transpose[col], row) << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<long> DictTransposeMatrix::sum(long axis) const {
    if (axis < 0 || axis > 1) {
        throw IndexOutOfBoundsException(axis, 2);
    }
    if (axis == 0) {  // sum across columns
        std::vector<long> totals(this->ncols, 0);
        for (long row_index = 0; row_index < this->nrows; ++row_index) {
            const MapVector<long> &row = this->matrix[row_index];
            for (const std::pair<long, long> &element : row) {
                totals[element.first] += totals[element.second];
            }
        }
        return totals;  // py::array_t<long>(this->ncols, totals);
    } else {  // (axis == 1) sum across rows
        std::vector<long> totals(this->nrows, 0);
        for (long row = 0; row < this->nrows; ++row) {
            const MapVector<long> &matrix_row = this->matrix[row];
            for (const std::pair<long, long> &element : matrix_row) {
                totals[row] += element.second;
            }
        }
        return totals;
    }
}

EdgeWeights DictTransposeMatrix::outgoing_edges(long block) const {
    check_row_bounds(block);
    std::vector<long> indices;
    std::vector<long> values;
    const MapVector<long> &block_row = this->matrix[block];
    for (const std::pair<long, long> &element : block_row) {
        indices.push_back(element.first);
        values.push_back(element.second);
    }
    return EdgeWeights {indices, values};
}

long DictTransposeMatrix::trace() const {
    long total = 0;
    // Assumes that the matrix is square (which it should be in this case)
    for (long index = 0; index < this->nrows; ++index) {
        total += this->get(index, index);
        // total += this->matrix[index][index];
    }
    return total;
}

void DictTransposeMatrix::update_edge_counts(long current_block, long proposed_block, std::vector<long> current_row,
                                             std::vector<long> proposed_row, std::vector<long> current_col,
                                             std::vector<long> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    for (long col = 0; col < ncols; ++col) {
        long current_val = current_row[col];
        if (current_val == 0) {
            this->matrix[current_block].erase(col);
            this->matrix_transpose[col].erase(current_block);
        } else {
            this->matrix[current_block][col] = current_val;
            this->matrix_transpose[col][current_block] = current_val;
        }
        long proposed_val = proposed_row[col];
        if (proposed_val == 0) {
            this->matrix[proposed_block].erase(col);
            this->matrix_transpose[col].erase(proposed_block);
        } else {
            this->matrix[proposed_block][col] = proposed_val;
            this->matrix_transpose[col][proposed_block] = proposed_val;  
        }
    }
    for (long row = 0; row < nrows; ++row) {
        long current_val = current_col[row];
        if (current_val == 0) {
            this->matrix[row].erase(current_block);
            this->matrix_transpose[current_block].erase(row);
        } else {
            this->matrix[row][current_block] = current_val;
            this->matrix_transpose[current_block][row] = current_val;
        }
        long proposed_val = proposed_col[row];
        if (proposed_val == 0) {
            this->matrix[row].erase(proposed_block);
            this->matrix_transpose[proposed_block].erase(row);
        } else {
            this->matrix[row][proposed_block] = proposed_val;
            this->matrix_transpose[proposed_block][row] = proposed_val;
        }
    }
}

void DictTransposeMatrix::update_edge_counts(long current_block, long proposed_block, MapVector<long> current_row,
                                             MapVector<long> proposed_row, MapVector<long> current_col,
                                             MapVector<long> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    for (long col = 0; col < ncols; ++col) {
        long current_val = current_row[col];
        if (current_val == 0) {
            this->matrix[current_block].erase(col);
            this->matrix_transpose[col].erase(current_block);
        } else {
            this->matrix[current_block][col] = current_val;
            this->matrix_transpose[col][current_block] = current_val;
        }
        long proposed_val = proposed_row[col];
        if (proposed_val == 0) {
            this->matrix[proposed_block].erase(col);
            this->matrix_transpose[col].erase(proposed_block);
        } else {
            this->matrix[proposed_block][col] = proposed_val;
            this->matrix_transpose[col][proposed_block] = proposed_val;
        }
    }
    for (long row = 0; row < nrows; ++row) {
        long current_val = current_col[row];
        if (current_val == 0) {
            this->matrix[row].erase(current_block);
            this->matrix_transpose[current_block].erase(row);
        } else {
            this->matrix[row][current_block] = current_val;
            this->matrix_transpose[current_block][row] = current_val;
        }
        long proposed_val = proposed_col[row];
        if (proposed_val == 0) {
            this->matrix[row].erase(proposed_block);
            this->matrix_transpose[proposed_block].erase(row);
        } else {
            this->matrix[row][proposed_block] = proposed_val;
            this->matrix_transpose[proposed_block][row] = proposed_val;
        }
    }
//    this->matrix[current_block] = MapVector<long>(current_row);
//    this->matrix[proposed_block] = MapVector<long>(proposed_row);
//    this->matrix_transpose[current_block] = MapVector<long>(current_col);
//    this->matrix_transpose[proposed_block] = MapVector<long>(proposed_col);
//    for (long block = 0; block < nrows; ++block) {
//        // TODO: try using get function (retrieve without modifying)
//        long current_val = current_col[block];  // matrix(block, current_block)
//        if (current_val == 0)
//            this->matrix[block].erase(current_block);
//        else
//            this->matrix[block][current_block] = current_val;
//        long proposed_val = proposed_col[block];  // matrix(block, proposed_block)
//        if (proposed_val == 0)
//            this->matrix[block].erase(proposed_block);
//        else
//            this->matrix[block][proposed_block] = proposed_val;
//        current_val = current_row[block];  // matrix(current_block, block)
//        if (current_val == 0)
//            this->matrix_transpose[block].erase(current_block);
//        else
//            this->matrix_transpose[block][current_block] = current_val;
//        proposed_val = proposed_row[block];  // matrix(proposed_block, block)
//        if (proposed_val == 0)
//            this->matrix_transpose[block].erase(proposed_block);
//        else
//            this->matrix_transpose[block][proposed_block] = current_val;
//    }

}

void DictTransposeMatrix::update_edge_counts(const Delta &delta) {
//    for (const std::pair<const std::pair<long, long>, long> &entry : delta) {
//        long row = entry.first.first;
//        long col = entry.first.second;
//        long change = entry.second;
    for (const std::tuple<long, long, long> &entry : delta.entries()) {
        long row = std::get<0>(entry);
        long col = std::get<1>(entry);
        long change = std::get<2>(entry);
        this->matrix[row][col] += change;
        this->matrix_transpose[col][row] += change;
        if (this->matrix[row][col] == 0) {
            this->matrix[row].erase(col);
            this->matrix_transpose[col].erase(row);
        }
    }
}

bool DictTransposeMatrix::validate(long row, long col, long val) const {
    long matrix_value = this->getrow_sparse(row)[col];
    long transpose_value = this->getcol_sparse(col)[row];
    if (val != matrix_value || val != transpose_value) {
        std::cout << "matrix[" << row << "," << col << "] = " << matrix_value << ", matrixT[" << col << "," << row
        << "] = " << transpose_value << " while actual value = " << val << std::endl;
        return false;
    }
    return true;
}

std::vector<long> DictTransposeMatrix::values() const {
    // TODO: maybe return a sparse vector every time?
    std::vector<long> values;
    for (long row = 0; row < nrows; ++row) {
        const MapVector<long> &matrix_row = this->matrix[row];
        for (const std::pair<const long, long> &element : matrix_row) {
            values.push_back(element.second);
        }
    }
    return values;
}
