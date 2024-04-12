#include "dict_matrix.hpp"

void DictMatrix::add(long row, long col, long val) {
    // TODO: bound checking includes branching, may get rid of it for performance
    check_row_bounds(row);
    check_col_bounds(col);
    matrix[row][col] += val;
}

void DictMatrix::clearcol(long col) {
    for (MapVector<long> &row : this->matrix)
        row.erase(col);
}

void DictMatrix::clearrow(long row) {
    this->matrix[row].clear();
}

// void DictMatrix::check_row_bounds(long row) {
//     if (row < 0 || row >= this->nrows) {
//         throw IndexOutOfBoundsException(row, this->nrows);
//     }
// }

// void DictMatrix::check_col_bounds(long col) {
//     if (col < 0 || col >= this->ncols) {
//         throw IndexOutOfBoundsException(col, this->ncols);
//     }
// }

ISparseMatrix* DictMatrix::copy() const {
    // TODO: this is probably inefficient (create a matrix, then use a copy constructor to return a new one)
    // std::vector<std::unordered_map<long, long>> dict_matrix(this->nrows, std::unordered_map<long, long>());
    ISparseMatrix* dict_matrix = new DictMatrix(this->nrows, this->ncols);
    for (long i = 0; i < this->nrows; ++i) {
        for (const std::pair<const long, long> &entry : this->matrix[i]) {
            dict_matrix->add(i, entry.first, entry.second);
        }
    }
    std::cout << "Returning copied dict_matrix" << std::endl;
    return dict_matrix;
//    DictMatrix dict_matrix(this->nrows, this->ncols);
//    for (long i = 0; i < this->nrows; ++i) {
//        const std::unordered_map<long, long> row = this->matrix[i];
//        dict_matrix.matrix[i] = row;  // TODO: double-check that this is a copy constructor
//    }
//    return new DictMatrix(dict_matrix);
}

long DictMatrix::distinct_edges(long block) const {
    long result = (long) this->matrix[block].size();
    for (long row = 0; row < this->nrows; ++row) {
        if (row == block) continue;  // Do not double-count self-edge
        const MapVector<long> &matrix_row = this->matrix[row];
        const auto iterator = matrix_row.find(block);
        if (iterator != matrix_row.end()) {
            result += 1;
        }
    }
    return result;
}

std::vector<std::tuple<long, long, long>> DictMatrix::entries() const {
    std::vector<std::tuple<long, long, long>> result;
    for (long row_index = 0; row_index < this->nrows; ++row_index) {
        const MapVector<long> &row = this->matrix[row_index];
        for (const std::pair<const long, long> &entry : row) {
            long col_index = entry.first;
            long value = entry.second;
            result.emplace_back(row_index, col_index, value);
        }
    }
    return result;
}

long DictMatrix::get(long row, long col) const {
    check_row_bounds(row);
    check_col_bounds(col);
    const MapVector<long> &row_vector = this->matrix[row];
    auto it = row_vector.find(col);
    if (it == row_vector.end())
        return 0;
    return it->second;
    // return matrix[row][col];
}

std::vector<long> DictMatrix::getcol(long col) const {
    check_col_bounds(col);
    std::vector<long> col_values(this->nrows, 0);
    for (long row = 0; row < this->nrows; ++row) {
//        const std::unordered_map<long, long> &matrix_row = this->matrix[row];
        const MapVector<long> &matrix_row = this->matrix[row];
        for (const std::pair<const long, long> &element : matrix_row) {
            if (element.first == col) {
                col_values[row] = element.second;
                break;
            }
        }
    }
    return col_values;
}

MapVector<long> DictMatrix::getcol_sparse(long col) const {
    check_col_bounds(col);
    MapVector<long> col_vector;
    for (long row = 0; row < this->nrows; ++row) {
        const MapVector<long> &matrix_row = this->matrix[row];
//        const std::unordered_map<long, long> &matrix_row = this->matrix[row];
        for (const std::pair<const long, long> &element : matrix_row) {
            if (element.first == col) {
                col_vector[row] = element.second;
                break;
            }
        }
    }
    return col_vector;
}

const MapVector<long>& DictMatrix::getcol_sparseref(long col) const {
    check_row_bounds(col);
    throw std::logic_error("sparse reference to DictMatrix column impossible to implement!");
//    return this->matrix_transpose[col];
}

void DictMatrix::getcol_sparse(long col, MapVector<long> &col_vector) const {
    check_col_bounds(col);
    for (long row = 0; row < this->nrows; ++row) {
        const MapVector<long> &matrix_row = this->matrix[row];
//        const std::unordered_map<long, long> &matrix_row = this->matrix[row];
        for (const std::pair<const long, long> &element : matrix_row) {
            if (element.first == col) {
                col_vector[row] = element.second;
                break;
            }
        }
    }
}

std::vector<long> DictMatrix::getrow(long row) const {
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
    // for (long col = 0; col < ncols; ++col) {
    //     row_values[col] = matrix[row][col];
    // }
    return row_values;  // py::array_t<long>(this->ncols, row_values);
}

MapVector<long> DictMatrix::getrow_sparse(long row) const {
    check_row_bounds(row);
    return this->matrix[row];
}

void DictMatrix::getrow_sparse(long row, MapVector<long> &col_vector) const {
    check_row_bounds(row);
    col_vector = this->matrix[row];
}

const MapVector<long>& DictMatrix::getrow_sparseref(long row) const {
    check_row_bounds(row);
    return this->matrix[row];
}

EdgeWeights DictMatrix::incoming_edges(long block) const {
    check_col_bounds(block);
    std::vector<long> indices;
    std::vector<long> values;
    for (long row = 0; row < this->nrows; ++row) {
        const MapVector<long> &matrix_row = this->matrix[row];
//        const std::unordered_map<long, long> &matrix_row = this->matrix[row];
        for (const std::pair<const long, long> &element : matrix_row) {
            if (element.first == block) {
                indices.push_back(row);
                values.push_back(element.second);
                break;
            }
        }
    }
    return EdgeWeights {indices, values};
}

std::set<long> DictMatrix::neighbors(long block) const {
    std::set<long> result;
    for (const std::pair<long, long> &entry : this->matrix[block]) {
        result.insert(entry.first);
    }
    for (long row = 0; row < this->nrows; ++row) {
        const MapVector<long> &matrix_row = this->matrix[row];
        const auto iterator = matrix_row.find(block);
        if (iterator != matrix_row.end()) {
            result.insert(row);
        }
    }
    return result;
}

MapVector<long> DictMatrix::neighbors_weights(long block) const {
    MapVector<long> result;
    for (const std::pair<long, long> &entry : this->matrix[block]) {
        result[entry.first] += entry.second;
    }
    for (long row = 0; row < this->nrows; ++row) {
        const MapVector<long> &matrix_row = this->matrix[row];
//        const std::unordered_map<long, long> &matrix_row = this->matrix[row];
        const auto iterator = matrix_row.find(block);
        if (iterator != matrix_row.end() && iterator->second != block) {
            result[iterator->first] += iterator->second;
//            result.insert(iterator->first);
        }
    }
    return result;
}

Indices DictMatrix::nonzero() const {
    std::vector<long> row_vector;
    std::vector<long> col_vector;
    for (long row = 0; row < nrows; ++row) {
        const MapVector<long> &matrix_row = this->matrix[row];
//        std::unordered_map<long, long> matrix_row = this->matrix[row];
        for (const std::pair<const long, long> &element : matrix_row) {
            row_vector.push_back(row);
            col_vector.push_back(element.first);
        }
        // for (long col = 0; col < ncols; ++col) {
        //     if (matrix(row, col) != 0) {
        //         row_vector.push_back(row);
        //         col_vector.push_back(col);
        //     }
        // }
    }
    return Indices{row_vector, col_vector};
}

EdgeWeights DictMatrix::outgoing_edges(long block) const {
    check_row_bounds(block);
    std::vector<long> indices;
    std::vector<long> values;
    const MapVector<long> &block_row = this->matrix[block];
//    const std::unordered_map<long, long> &block_row = this->matrix[block];
    for (const std::pair<const long, long> &element : block_row) {
        indices.push_back(element.first);
        values.push_back(element.second);
    }
    return EdgeWeights {indices, values};
}

void DictMatrix::setrow(long row, const MapVector<long> &vector) {
    check_row_bounds(row);
    this->matrix[row] = MapVector<long>(vector);
}

void DictMatrix::setcol(long col, const MapVector<long> &vector) {
    check_col_bounds(col);
    for (long row = 0; row < (long) this->matrix.size(); ++row) {
        MapVector<long>::const_iterator value = vector.find(row);
        if (value == vector.end())  // value is not in vector
            this->matrix[row].erase(col);
        else
            this->matrix[row][col] = value->second;
    }
}

void DictMatrix::sub(long row, long col, long val) {
    check_row_bounds(row);
    check_col_bounds(col);
    // TODO: debug mode - if matrix[row][col] doesn't exist, throw exception
    matrix[row][col] -= val;
}

long DictMatrix::edges() const {
    long total = 0;
    for (long row = 0; row < nrows; ++row) {
        const MapVector<long> &matrix_row = this->matrix[row];
//        const std::unordered_map<long, long> &matrix_row = this->matrix[row];
        for (const std::pair<const long, long> &element : matrix_row) {
            total += element.second;
        }
        // for (long col = 0; col < ncols; ++col) {
        //     total += matrix(row, col);
        // }
    }
    return total;
}

void DictMatrix::print() const {
    for (long row = 0; row < this->nrows; ++row) {
        for (long col = 0; col < this->ncols; ++col) {
            std::cout << this->get(row, col) << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<long> DictMatrix::sum(long axis) const {
    if (axis < 0 || axis > 1) {
        throw IndexOutOfBoundsException(axis, 2);
    }
    if (axis == 0) {  // sum across columns
        std::vector<long> totals(this->ncols, 0);
        for (long row_index = 0; row_index < this->nrows; ++row_index) {
        // for (const std::unordered_map<long, long> &row : this->matrix) {
            const MapVector<long> &row = this->matrix[row_index];
//            const std::unordered_map<long, long> &row = this->matrix[row_index];
            for (const std::pair<const long, long> &element : row) {
                totals[element.first] += totals[element.second];
            }
        }
        // for (long row = 0; row < this->nrows; ++row) {
        //     for (long col = 0; col < this->ncols; ++col) {
        //         totals[col] += this->matrix(row, col);
        //     }
        // }
        return totals;  // py::array_t<long>(this->ncols, totals);
    } else {  // (axis == 1) sum across rows
        std::vector<long> totals(this->nrows, 0);
        for (long row = 0; row < this->nrows; ++row) {
            const MapVector<long> &matrix_row = this->matrix[row];
//            const std::unordered_map<long, long> &matrix_row = this->matrix[row];
            for (const std::pair<const long, long> &element : matrix_row) {
                totals[row] += element.second;
            }
            // for (long col = 0; col < this->ncols; ++col) {
            //     totals[row] += this->matrix(row, col);
            // }
        }
        return totals;
    }
}

long DictMatrix::trace() const {
    long total = 0;
    // Assumes that the matrix is square (which it should be in this case)
    for (long index = 0; index < this->nrows; ++index) {
        // TODO: this creates 0 elements where they don't exist. To optimize memory, could add a find call first
        total += this->get(index, index);
        // total += this->matrix[index][index];
    }
    return total;
}

void DictMatrix::update_edge_counts(long current_block, long proposed_block, std::vector<long> current_row,
    std::vector<long> proposed_row, std::vector<long> current_col, std::vector<long> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    for (long col = 0; col < ncols; ++col) {
        long current_val = current_row[col];
        if (current_val == 0)
            this->matrix[current_block].erase(col);
        else
            this->matrix[current_block][col] = current_val;
        long proposed_val = proposed_row[col];
        if (proposed_val == 0)
            this->matrix[proposed_block].erase(col);
        else
            this->matrix[proposed_block][col] = proposed_val;
    }
    for (long row = 0; row < nrows; ++row) {
        long current_val = current_col[row];
        if (current_val == 0)
            this->matrix[row].erase(current_block);
        else
            this->matrix[row][current_block] = current_val;
        long proposed_val = proposed_col[row];
        if (proposed_val == 0)
            this->matrix[row].erase(proposed_block);
        else
            this->matrix[row][proposed_block] = proposed_val;
    }
}

void DictMatrix::update_edge_counts(long current_block, long proposed_block, MapVector<long> current_row,
                                    MapVector<long> proposed_row, MapVector<long> current_col,
                                    MapVector<long> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    this->matrix[current_block] = MapVector<long>(current_row);
    this->matrix[proposed_block] = MapVector<long>(proposed_row);
    for (long row = 0; row < nrows; ++row) {
        long current_val = current_col[row];
        if (current_val == 0)
            this->matrix[row].erase(current_block);
        else
            this->matrix[row][current_block] = current_val;
        long proposed_val = proposed_col[row];
        if (proposed_val == 0)
            this->matrix[row].erase(proposed_block);
        else
            this->matrix[row][proposed_block] = proposed_val;
    }
}

void DictMatrix::update_edge_counts(const Delta &delta) {
    for (const std::tuple<long, long, long> &entry : delta.entries()) {
        long row = std::get<0>(entry);
        long col = std::get<1>(entry);
        long change = std::get<2>(entry);
        this->matrix[row][col] += change;
        if (this->matrix[row][col] == 0) {
            this->matrix[row].erase(col);
        }
    }
}

bool DictMatrix::validate(long row, long col, long val) const {
    long value = this->get(row, col);
    return val == value;
}

std::vector<long> DictMatrix::values() const {
    // TODO: maybe return a sparse vector every time?
    std::vector<long> values;
    for (long row = 0; row < nrows; ++row) {
        const MapVector<long> &matrix_row = this->matrix[row];
//        const std::unordered_map<long, long> &matrix_row = this->matrix[row];
        for (const std::pair<const long, long> &element : matrix_row) {
            values.push_back(element.second);
        }
        // for (long col = 0; col < ncols; ++col) {
        //     long value = matrix(row, col);
        //     if (value != 0) {
        //         values.push_back(value);
        //     }
        // }
    }
    return values;
}
