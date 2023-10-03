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
#include "boost_mapped_matrix.hpp"

void BoostMappedMatrix::add(long row, long col, long val) {
    // TODO: bound checking includes branching, may get rid of it for performance
    check_row_bounds(row);
    check_col_bounds(col);
    matrix(row, col) += val;
}

void BoostMappedMatrix::check_row_bounds(long row) {
    if (row < 0 || row >= this->nrows) {
        throw IndexOutOfBoundsException(row, this->nrows);
    }
}

void BoostMappedMatrix::check_col_bounds(long col) {
    if (col < 0 || col >= this->ncols) {
        throw IndexOutOfBoundsException(col, this->ncols);
    }
}

BoostMappedMatrix BoostMappedMatrix::copy() {
    BoostMappedMatrix boost_mapped_matrix(this->nrows, this->ncols);
    boost_mapped_matrix.matrix = boost::numeric::ublas::mapped_matrix<long>(this->matrix);
    return boost_mapped_matrix;
}

long BoostMappedMatrix::get(long row, long col) {
    check_row_bounds(row);
    check_col_bounds(col);
    return matrix(row, col);
}

std::vector<long> BoostMappedMatrix::getrow(long row) {
    check_row_bounds(row);
    std::vector<long> row_values = utils::constant<long>(this->ncols, 0);
    // long row_values [this->ncols];
    for (long col = 0; col < ncols; ++col) {
        row_values[col] = matrix(row, col);
    }
    return row_values;  // py::array_t<long>(this->ncols, row_values);
}

std::vector<long> BoostMappedMatrix::getcol(long col) {
    // check_col_bounds(col);
    // std::vector<long> col_values = utils::constant<long>(this->nrows, 0);
    std::vector<long> col_values(this->nrows, 0);
    // long col_values [this->nrows];
    for (long row = 0; row < nrows; ++row) {
        col_values[row] = matrix(row, col);
    }
    return col_values;  // py::array_t<long>(this->nrows, col_values);
}

EdgeWeights BoostMappedMatrix::incoming_edges(long block) {
    check_col_bounds(block);
    std::vector<long> indices;
    std::vector<long> values;
    for (long row = 0; row < this->nrows; ++row) {
        long value = this->matrix(row, block);
        if (value != 0) {
            indices.push_back(row);
            values.push_back(value);
        } else {
            this->matrix.erase_element(row, block);
        }
    }
    return EdgeWeights {indices, values};
}

Indices BoostMappedMatrix::nonzero() {
    std::vector<long> row_vector;
    std::vector<long> col_vector;
    for (long row = 0; row < nrows; ++row) {
        for (long col = 0; col < ncols; ++col) {
            if (matrix(row, col) != 0) {
                row_vector.push_back(row);
                col_vector.push_back(col);
            }
        }
    }
    return Indices{row_vector, col_vector};
}

void BoostMappedMatrix::sub(long row, long col, long val) {
    check_row_bounds(row);
    check_col_bounds(col);
    matrix(row, col) -= val;
}

long BoostMappedMatrix::sum() {
    long total = 0;
    for (long row = 0; row < nrows; ++row) {
        for (long col = 0; col < ncols; ++col) {
            total += matrix(row, col);
        }
    }
    return total;
}

std::vector<long> BoostMappedMatrix::sum(long axis) {
    if (axis < 0 || axis > 1) {
        throw IndexOutOfBoundsException(axis, 2);
    }
    if (axis == 0) {  // sum across columns
        std::vector<long> totals = utils::constant<long>(this->ncols, 0);
        for (long row = 0; row < this->nrows; ++row) {
            for (long col = 0; col < this->ncols; ++col) {
                totals[col] += this->matrix(row, col);
            }
        }
        return totals;  // py::array_t<long>(this->ncols, totals);
    } else {  // (axis == 1) sum across rows
        std::vector<long> totals = utils::constant<long>(this->nrows, 0);
        for (long row = 0; row < this->nrows; ++row) {
            for (long col = 0; col < this->ncols; ++col) {
                totals[row] += this->matrix(row, col);
            }
        }
        return totals;
    }
}

long BoostMappedMatrix::trace() {
    long total = 0;
    // Assumes that the matrix is square (which it should be in this case)
    for (long index = 0; index < this->nrows; ++index) {
        total += this->matrix(index, index);
    }
    return total;
}

// long BoostMappedMatrix::operator[] (py::tuple index) {
//     py::array_t<long> tuple_array(index);
//     auto tuple_vals = tuple_array.mutable_unchecked<1>();
//     long row = tuple_vals(0);
//     long col = tuple_vals(1);
//     return this->matrix(row, col);
// }

EdgeWeights BoostMappedMatrix::outgoing_edges(long block) {
    check_row_bounds(block);
    std::vector<long> indices;
    std::vector<long> values;
    for (long col = 0; col < this->ncols; ++col) {
        long value = this->matrix(block, col);
        if (value != 0) {
            indices.push_back(col);
            values.push_back(value);
        } else {
            this->matrix.erase_element(block, col);
        }
    }
    return EdgeWeights {indices, values};
}

void BoostMappedMatrix::update_edge_counts(long current_block, long proposed_block, std::vector<long> current_row,
    std::vector<long> proposed_row, std::vector<long> current_col, std::vector<long> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    for (long col = 0; col < ncols; ++col) {
        long current_val = current_row[col];
        if (current_val == 0)
            this->matrix.erase_element(current_block, col);
        else
            this->matrix(current_block, col) = current_val;
        long proposed_val = proposed_row[col];
        if (proposed_val == 0)
            this->matrix.erase_element(proposed_block, col);
        else
            this->matrix(proposed_block, col) = proposed_val;
    }
    for (long row = 0; row < nrows; ++row) {
        long current_val = current_col[row];
        if (current_val == 0)
            this->matrix.erase_element(row, current_block);
        else
            this->matrix(row, current_block) = current_val;
        long proposed_val = proposed_col[row];
        if (proposed_val == 0)
            this->matrix.erase_element(row, proposed_block);
        else
            this->matrix(row, proposed_block) = proposed_val;
    }
}

std::vector<long> BoostMappedMatrix::values() {
    // TODO: maybe return a sparse vector every time?
    std::vector<long> values;
    for (long row = 0; row < nrows; ++row) {
        for (long col = 0; col < ncols; ++col) {
            long value = matrix(row, col);
            if (value != 0) {
                values.push_back(value);
            }
        }
    }
    return values;
}
