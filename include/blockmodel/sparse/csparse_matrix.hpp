/***
 * Common interface for sparse matrix types.
 */
#ifndef CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP

#include <exception>
#include <iostream>
#include <memory>
#include <sstream>

// #include <Eigen/Core>
#include "delta.hpp"
#include "typedefs.hpp"

// typedef Eigen::VectorXi Vector;

typedef struct indices_t {
    std::vector<long> rows;
    std::vector<long> cols;
} Indices;

class IndexOutOfBoundsException: public std::exception {
public:
    IndexOutOfBoundsException(long index, long max) { // } : index(index), max(max) {
        std::ostringstream message_stream;
        message_stream << "Index " << index << " is out of bounds [0, " << max - 1 << "]";
        this->message = message_stream.str();
    }
    const char* what() const noexcept override {
        return this->message.c_str();
    }
private:
//    long index;
//    long max;
    std::string message;
};

///
/// C++ implementation of the sparse matrix interface
///
class ISparseMatrix {
public:
    // ISparseMatrix() {}
    virtual ~ISparseMatrix() = default;
    /// Add `val` to `matrix[row, col]`.
    virtual void add(long row, long col, long val) = 0;
    // virtual void add(long row, std::vector<long> cols, std::vector<long> values) = 0;
    /// Set matrix row `row` to empty.
    virtual void clearrow(long row) = 0;
    /// Set matrix column `col` to empty.
    virtual void clearcol(long col) = 0;
    /// Returns a copy of this matrix.
    [[nodiscard]] virtual ISparseMatrix* copy() const = 0;
    /// Returns the number of distinct block-level edges of `block`. Directed edges between two blocks are counted
    /// separately.
    [[nodiscard]] virtual long distinct_edges(long block) const = 0;
    /// Returns matrix entries in the form `std::tuple<long, long, long`.
    [[nodiscard]] virtual std::vector<std::tuple<long, long, long>> entries() const = 0;
    /// Returns the value in `matrix[row, col]`.
    [[nodiscard]] virtual long get(long row, long col) const = 0;
    /// Returns the column `col` as a dense vector.
    [[nodiscard]] virtual std::vector<long> getcol(long col) const = 0;
    /// Returns the column `col` as a sparse vector.
    [[nodiscard]] virtual MapVector<long> getcol_sparse(long col) const = 0;
    /// Returns the col `col` as a reference to a sparse vector.
    [[nodiscard]] virtual const MapVector<long>& getcol_sparseref(long col) const = 0;
    /// Populates the values in `col_vector` with the values of column `col`.
    virtual void getcol_sparse(long col, MapVector<long> &col_vector) const = 0;
    // virtual const MapVector<long>& getcol_sparse(long col) const = 0;
    /// Returns the row `row` as a dense vector.
    [[nodiscard]] virtual std::vector<long> getrow(long row) const = 0;
    /// Returns the row `row` as a sparse vector.
    [[nodiscard]] virtual MapVector<long> getrow_sparse(long row) const = 0;
    /// Returns the row `row` as a reference to a sparse vector.
    [[nodiscard]] virtual const MapVector<long>& getrow_sparseref(long row) const = 0;
    /// Populates the values in `row_vector` with the values of row `row`.
    virtual void getrow_sparse(long row, MapVector<long> &row_vector) const = 0;
    // virtual const MapVector<long>& getrow_sparse(long row) const = 0;
    /// TODO: docstring
    [[nodiscard]] virtual EdgeWeights incoming_edges(long block) const = 0;
    /// Returns the set of all neighbors of `block`. This includes `block` if it has self-edges.
    [[nodiscard]] virtual std::set<long> neighbors(long block) const = 0;
    /// Returns the set of weighted neighbors of `block`. This includes `block` if it has self-edges.
    [[nodiscard]] virtual MapVector<long> neighbors_weights(long block) const = 0;
    /// TODO: docstring
    [[nodiscard]] virtual Indices nonzero() const = 0;
    /// TODO: docstring
    [[nodiscard]] virtual EdgeWeights outgoing_edges(long block) const = 0;
    /// Sets the values in a row equal to the input vector `vector`.
    virtual void setrow(long row, const MapVector<long> &vector) = 0;
    /// Sets the values in a column equal to the input vector `vector`.
    virtual void setcol(long col, const MapVector<long> &vector) = 0;
    /// Subtracts `val` from `matrix[row, col]`.
    virtual void sub(long row, long col, long val) = 0;
    [[nodiscard]] virtual long edges() const = 0;
    virtual void print() const = 0;
    [[nodiscard]] virtual std::vector<long> sum(long axis = 0) const = 0;
    [[nodiscard]] virtual long trace() const = 0;
    /// Updates the blockmatrix values in the rows and columns corresponding to `current_block` and `proposed_block`.
    virtual void update_edge_counts(long current_block, long proposed_block, std::vector<long> current_row,
                                    std::vector<long> proposed_row, std::vector<long> current_col,
                                    std::vector<long> proposed_col) = 0;
    /// Updates the blockmatrix values in the rows and columns corresponding to `current_block` and `proposed_block`.
    virtual void update_edge_counts(long current_block, long proposed_block, MapVector<long> current_row,
                                    MapVector<long> proposed_row, MapVector<long> current_col,
                                    MapVector<long> proposed_col) = 0;
    /// Updates the blockmatrix values using the changes to the blockmodel stored in `delta`.
    virtual void update_edge_counts(const Delta &delta) = 0;
    /// Returns true if the value in matrix[`row`, `col`] == `val`.
    [[nodiscard]] virtual bool validate(long row, long col, long val) const = 0;
    [[nodiscard]] virtual std::vector<long> values() const = 0;
    std::pair<long, long> shape;

protected:
    void check_row_bounds(long row) const {
        if (row < 0 || row >= this->nrows) {
            throw IndexOutOfBoundsException(row, this->nrows);
        }
    }
    void check_col_bounds(long col) const {
        if (col < 0 || col >= this->ncols) {
            throw IndexOutOfBoundsException(col, this->ncols);
        }
    }
    long ncols;
    long nrows;
};

///
/// C++ implementation of the distributed sparse matrix interface
///
class IDistSparseMatrix : public ISparseMatrix {
public:
    // IDistSparseMatrix() {}
    virtual ~IDistSparseMatrix() {}
    /// Returns true if this process owns this block.
    virtual bool stores(long block) const = 0;
    /// Returns a copy of this distributed matrix.
    virtual IDistSparseMatrix* copyDistSparseMatrix() const = 0;

protected:
    std::vector<long> _ownership;
    virtual void sync_ownership(const std::vector<long> &myblocks) = 0;

};

// typedef std::unique_ptr<ISparseMatrix> SparseMatrix;

#endif // CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP
