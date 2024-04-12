#include "distributed/dist_dict_matrix.hpp"

void DistDictMatrix::add(long row, long col, long val) {
    // TODO: bound checking includes branching, may get rid of it for performance
    check_row_bounds(row);
    check_col_bounds(col);
    this->_matrix[row][col] += val;
}

void DistDictMatrix::clearcol(long col) {
    for (MapVector<long> &row : this->_matrix)
        row.erase(col);
}

void DistDictMatrix::clearrow(long row) {
    this->_matrix[row].clear();
}

ISparseMatrix* DistDictMatrix::copy() const {
    return this->copyDistSparseMatrix();
}

IDistSparseMatrix* DistDictMatrix::copyDistSparseMatrix() const {
    DistDictMatrix *matrix = new DistDictMatrix();
    matrix->nrows = this->nrows;
    matrix->ncols = this->ncols;
    for (long i = 0; i < this->nrows; ++i) {
        const std::unordered_map<long, long> row = this->_matrix[i];
        matrix->_matrix[i] = row;  // TODO: double-check that this is a copy constructor
    }
    matrix->_ownership = std::vector<long>(this->_ownership);
    return matrix;
}

long DistDictMatrix::get(long row, long col) const {
    check_row_bounds(row);
    check_col_bounds(col);
    const MapVector<long> &row_vector = this->_matrix[row];
    auto it = row_vector.find(col);
    if (it == row_vector.end())
        return 0;
    return it->second;
}

std::vector<long> DistDictMatrix::getcol(long col) const {
    check_col_bounds(col);
    std::vector<long> col_values(this->nrows, 0);
    for (long row = 0; row < this->nrows; ++row) {
        const std::unordered_map<long, long> &matrix_row = this->_matrix[row];
        for (const std::pair<long, long> &element : matrix_row) {
            if (element.first == col) {
                col_values[row] = element.second;
                break;
            }
        }
    }
    return col_values;
}

MapVector<long> DistDictMatrix::getcol_sparse(long col) const {
    check_col_bounds(col);
    MapVector<long> col_vector;
    for (long row = 0; row < this->nrows; ++row) {
        const std::unordered_map<long, long> &matrix_row = this->_matrix[row];
        for (const std::pair<long, long> &element : matrix_row) {
            if (element.first == col) {
                col_vector[row] = element.second;
                break;
            }
        }
    }
    return col_vector;
}

void DistDictMatrix::getcol_sparse(long col, MapVector<long> &col_vector) const {
    check_col_bounds(col);
    for (long row = 0; row < this->nrows; ++row) {
        const std::unordered_map<long, long> &matrix_row = this->_matrix[row];
        for (const std::pair<long, long> &element : matrix_row) {
            if (element.first == col) {
                col_vector[row] = element.second;
                break;
            }
        }
    }
}

std::vector<long> DistDictMatrix::getrow(long row) const {
    throw "Dense getrow used!";
    check_row_bounds(row);
    if (this->stores(row)) {
        std::vector<long> row_values = utils::constant<long>(this->ncols, 0);
        const MapVector<long> &matrix_row = this->_matrix[row];
        for (const std::pair<long, long> element : matrix_row) {
            row_values[element.first] = element.second;
        }
        return row_values;  // py::array_t<long>(this->ncols, row_values);
    }
    // send message asking for row
    MPI_Send(&row, 1, MPI_LONG, this->_ownership[row], MSG_GETROW + omp_get_thread_num(), mpi.comm);
    long rowsize = -1;
    MPI_Status status;
    MPI_Recv(&rowsize, 1, MPI_LONG, this->_ownership[row], MSG_SIZEROW + omp_get_thread_num(), mpi.comm, &status);
    // Other side
    MPI_Status status2;
    long requested;
    MPI_Recv(&requested, 1, MPI_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, mpi.comm, &status2);
    long threadID = status2.MPI_TAG % 100000;
    long tag = status2.MPI_TAG - threadID;
    if (tag == MSG_GETROW) {
        auto row = this->getrow(requested);
        // MPI_Send(&(row.data()), )
    }
}

MapVector<long> DistDictMatrix::getrow_sparse(long row) const {
    throw "Wrong sparse getrow used!";
    check_row_bounds(row);
    return this->_matrix[row];
}

void DistDictMatrix::getrow_sparse(long row, MapVector<long> &row_vector) const {
    check_row_bounds(row);
    if (this->stores(row)) {
        row_vector = this->_matrix[row];
        return;
    }
    // send message asking for row
    MPI_Send(&row, 1, MPI_LONG, this->_ownership[row], MSG_GETROW + omp_get_thread_num(), mpi.comm);
    long rowsize = -1;
    MPI_Status status;
    MPI_Recv(&rowsize, 1, MPI_LONG, this->_ownership[row], MSG_SIZEROW + omp_get_thread_num(), mpi.comm, &status);
    // Other side
    MPI_Status status2;
    long requested;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, mpi.comm, &status2);
    // Change count type depending on status
    MPI_Get_count(&status, MPI_LONG, &requested);
    MPI_Recv(&requested, 1, MPI_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, mpi.comm, &status2);
    long threadID = status2.MPI_TAG % 100000;
    long tag = status2.MPI_TAG - threadID;
    if (tag == MSG_GETROW) {
        MapVector<long> reqrow;
        std::vector<long> sendrow;
        this->getrow_sparse(requested, reqrow);
        for (auto p : reqrow) {
            sendrow.push_back(p.first);
            sendrow.push_back(p.second);
        }
        MPI_Send(sendrow.data(), sendrow.size(), MPI_LONG, status2.MPI_SOURCE, MSG_SENDROW, mpi.comm);
    }
}

EdgeWeights DistDictMatrix::incoming_edges(long block) const {
    check_col_bounds(block);
    std::vector<long> indices;
    std::vector<long> values;
    for (long row = 0; row < this->nrows; ++row) {
        const std::unordered_map<long, long> &matrix_row = this->_matrix[row];
        for (const std::pair<long, long> &element : matrix_row) {
            if (element.first == block) {
                indices.push_back(row);
                values.push_back(element.second);
                break;
            }
        }
    }
    return EdgeWeights {indices, values};
}

Indices DistDictMatrix::nonzero() const {
    std::vector<long> row_vector;
    std::vector<long> col_vector;
    for (long row = 0; row < nrows; ++row) {
        std::unordered_map<long, long> matrix_row = this->_matrix[row];
        for (const std::pair<long, long> &element : matrix_row) {
            row_vector.push_back(row);
            col_vector.push_back(element.first);
        }
    }
    return Indices{row_vector, col_vector};
}

EdgeWeights DistDictMatrix::outgoing_edges(long block) const {
    check_row_bounds(block);
    std::vector<long> indices;
    std::vector<long> values;
    const std::unordered_map<long, long> &block_row = this->_matrix[block];
    for (const std::pair<long, long> &element : block_row) {
        indices.push_back(element.first);
        values.push_back(element.second);
    }
    return EdgeWeights {indices, values};
}

bool DistDictMatrix::stores(long block) const {
    return this->_ownership[block] == mpi.rank;
}

void DistDictMatrix::setrow(long row, const MapVector<long> &vector) {
    check_row_bounds(row);
    this->_matrix[row] = MapVector<long>(vector);
}

void DistDictMatrix::setcol(long col, const MapVector<long> &vector) {
    check_col_bounds(col);
    for (long row = 0; row < (long) this->_matrix.size(); ++row) {
        MapVector<long>::const_iterator value = vector.find(row);
        if (value == vector.end())  // value is not in vector
            this->_matrix[row].erase(col);
        else
            this->_matrix[row][col] = value->second;
    }
}

void DistDictMatrix::sub(long row, long col, long val) {
    check_row_bounds(row);
    check_col_bounds(col);
    // TODO: debug mode - if matrix[row][col] doesn't exist, throw exception
    _matrix[row][col] -= val;
}

long DistDictMatrix::edges() const {
    long total = 0;
    for (long row = 0; row < nrows; ++row) {
        const std::unordered_map<long, long> &matrix_row = this->_matrix[row];
        for (const std::pair<long, long> &element : matrix_row) {
            total += element.second;
        }
    }
    return total;
}

std::vector<long> DistDictMatrix::sum(long axis) const {
    if (axis < 0 || axis > 1) {
        throw IndexOutOfBoundsException(axis, 2);
    }
    if (axis == 0) {  // sum across columns
        std::vector<long> totals(this->ncols, 0);
        for (long row_index = 0; row_index < this->nrows; ++row_index) {
            const std::unordered_map<long, long> &row = this->_matrix[row_index];
            for (const std::pair<long, long> &element : row) {
                totals[element.first] += totals[element.second];
            }
        }
        return totals;
    } else {  // (axis == 1) sum across rows
        std::vector<long> totals(this->nrows, 0);
        for (long row = 0; row < this->nrows; ++row) {
            const std::unordered_map<long, long> &matrix_row = this->_matrix[row];
            for (const std::pair<long, long> &element : matrix_row) {
                totals[row] += element.second;
            }
        }
        return totals;
    }
}


void DistDictMatrix::sync_ownership(const std::vector<long> &myblocks) {
    long numblocks[mpi.num_processes];
    long num_blocks = myblocks.size();
    MPI_Allgather(&(num_blocks), 1, MPI_LONG, &numblocks, 1, MPI_LONG, mpi.comm);
    long offsets[mpi.num_processes];
    offsets[0] = 0;
    for (long i = 1; i < mpi.num_processes; ++i) {
        offsets[i] = offsets[i-1] + numblocks[i-1];
    }
    long global_num_blocks = offsets[mpi.num_processes-1] + numblocks[mpi.num_processes-1];
    this->_ownership = std::vector<long>(global_num_blocks, -1);
    std::cout << "rank: " << mpi.rank << " num_blocks: " << num_blocks << " and globally: " << global_num_blocks << std::endl;
    std::vector<long> allblocks(global_num_blocks, -1);
    MPI_Allgatherv(myblocks.data(), num_blocks, MPI_LONG, allblocks.data(), &(numblocks[0]), &(offsets[0]), MPI_LONG, mpi.comm);
    if (mpi.rank == 0) {
        utils::print<long>(allblocks);
    }
    long owner = 0;
    for (long i = 0; i < global_num_blocks; ++i) {
        if (owner < mpi.num_processes - 1 && i >= offsets[owner+1]) {
            owner++;
        }
        this->_ownership[allblocks[i]] = owner;
    }
    if (mpi.rank == 0) {
        utils::print<long>(this->_ownership);
    }
}

long DistDictMatrix::trace() const {
    long total = 0;
    // Assumes that the matrix is square (which it should be in this case)
    for (long index = 0; index < this->nrows; ++index) {
        // TODO: this creates 0 elements where they don't exist. To optimize memory, could add a find call first
        total += this->get(index, index);
    }
    return total;
}

void DistDictMatrix::update_edge_counts(long current_block, long proposed_block, std::vector<long> current_row,
    std::vector<long> proposed_row, std::vector<long> current_col, std::vector<long> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    for (long col = 0; col < ncols; ++col) {
        long current_val = current_row[col];
        if (current_val == 0)
            this->_matrix[current_block].erase(col);
        else
            this->_matrix[current_block][col] = current_val;
        long proposed_val = proposed_row[col];
        if (proposed_val == 0)
            this->_matrix[proposed_block].erase(col);
        else
            this->_matrix[proposed_block][col] = proposed_val;
    }
    for (long row = 0; row < nrows; ++row) {
        long current_val = current_col[row];
        if (current_val == 0)
            this->_matrix[row].erase(current_block);
        else
            this->_matrix[row][current_block] = current_val;
        long proposed_val = proposed_col[row];
        if (proposed_val == 0)
            this->_matrix[row].erase(proposed_block);
        else
            this->_matrix[row][proposed_block] = proposed_val;
    }
}

void DistDictMatrix::update_edge_counts(const PairIndexVector &delta) {
    for (const std::pair<const std::pair<long, long>, long> &entry : delta) {
        long row = entry.first.first;
        long col = entry.first.second;
        long change = entry.second;
        this->_matrix[row][col] += change;
        if (this->_matrix[row][col] == 0)
            this->_matrix[row].erase(col);
    }
}

std::vector<long> DistDictMatrix::values() const {
    // TODO: maybe return a sparse vector every time?
    std::vector<long> values;
    for (long row = 0; row < nrows; ++row) {
        const std::unordered_map<long, long> &matrix_row = this->_matrix[row];
        for (const std::pair<long, long> &element : matrix_row) {
            values.push_back(element.second);
        }
    }
    return values;
}
