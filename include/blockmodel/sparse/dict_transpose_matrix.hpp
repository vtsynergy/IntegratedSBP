/***
 * Sparse Matrix that uses a vector of unordered maps to store the blockmodel.
 */
#ifndef CPPSBP_PARTITION_SPARSE_DICT_TRANSPOSE_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_DICT_TRANSPOSE_MATRIX_HPP

#include <map>
#include <unordered_map>

#include "csparse_matrix.hpp"
// TODO: figure out where to put utils.hpp so this never happens
#include "delta.hpp"
#include "typedefs.hpp"
#include "../../utils.hpp"

// #include <Eigen/Core>

/**
 * The list-of-maps sparse matrix, with a transpose for faster column indexing.
 * TODO: figure out where 0s are being added to the matrix, and whether or not we need to get rid of that
 */
class DictTransposeMatrix : public ISparseMatrix {
  public:
    DictTransposeMatrix() = default;
    DictTransposeMatrix(long nrows, long ncols, long buckets = 10) {
        this->ncols = ncols;
        this->nrows = nrows;
        this->matrix = std::vector<MapVector<long>>(this->nrows, MapVector<long>(buckets));
        this->matrix_transpose = std::vector<MapVector<long>>(this->ncols, MapVector<long>(buckets));
        this->shape = std::make_pair(this->nrows, this->ncols);
    }
    void add(long row, long col, long val) override;
    void add_transpose(long row, long col, long val);
    // void add(long row, std::vector<long> cols, std::vector<long> values) override;
    /// Clears the value in a given row. Complexity ~O(number of blocks).
    void clearrow(long row) override;
    /// Clears the values in a given column. Complexity ~O(number of blocks).
    void clearcol(long col) override;
    /// Returns a copy of the current matrix.
    ISparseMatrix* copy() const override;
    long distinct_edges(long block) const override;
    std::vector<std::tuple<long, long, long>> entries() const override;
    long get(long row, long col) const override;
    /// Returns all values in the requested column as a dense vector.
    std::vector<long> getcol(long col) const override;
    /// Returns all values in the requested column as a sparse vector (ordered map).
    MapVector<long> getcol_sparse(long col) const override;
    const MapVector<long>& getcol_sparseref(long col) const override;
    void getcol_sparse(long col, MapVector<long> &col_vector) const override;
    /// Returns all values in the requested row as a dense vector.
    std::vector<long> getrow(long row) const override;
    /// Returns all values in the requested column as a sparse vector (ordered map).
    MapVector<long> getrow_sparse(long row) const override;
    const MapVector<long>& getrow_sparseref(long row) const override;
    // const MapVector<long>& getrow_sparse(long row) const;
    void getrow_sparse(long row, MapVector<long> &row_vector) const override;
    EdgeWeights incoming_edges(long block) const override;
    std::set<long> neighbors(long block) const override;
    MapVector<long> neighbors_weights(long block) const override;
    Indices nonzero() const override;
    EdgeWeights outgoing_edges(long block) const override;
    /// Sets the values in a row equal to the input vector.
    void setrow(long row, const MapVector<long> &vector) override;
    /// Sets the values in a column equal to the input vector.
    void setcol(long col, const MapVector<long> &vector) override;
    void sub(long row, long col, long val) override;
    long edges() const override;
    void print() const override;
    std::vector<long> sum(long axis = 0) const override;
    long trace() const override;
    void update_edge_counts(long current_block, long proposed_block, std::vector<long> current_row,
                                    std::vector<long> proposed_row, std::vector<long> current_col,
                                    std::vector<long> proposed_col) override;
    void update_edge_counts(long current_block, long proposed_block, MapVector<long> current_row,
                            MapVector<long> proposed_row, MapVector<long> current_col,
                            MapVector<long> proposed_col) override;
    void update_edge_counts(const Delta &delta) override;
    bool validate(long row, long col, long val) const override;
    std::vector<long> values() const override;

  private:
    // void check_row_bounds(long row);
    // void check_col_bounds(long col);
    // void check_row_bounds(long row) const;
    // void check_col_bounds(long col) const;
    // long ncols;
    // long nrows;
//    std::vector<std::unordered_map<long, long>> matrix;
//    std::vector<std::unordered_map<long, long>> matrix_transpose;
    std::vector<MapVector<long>> matrix;
    std::vector<MapVector<long>> matrix_transpose;
};

#endif // CPPSBP_PARTITION_SPARSE_DICT_TRANSPOSE_MATRIX_HPP
