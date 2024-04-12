/***
 * Sparse Matrix that uses a vector of unordered maps to store the blockmodel.
 */
#ifndef CPPSBP_PARTITION_SPARSE_DICT_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_DICT_MATRIX_HPP

#include <unordered_map>

#include "csparse_matrix.hpp"
#include "delta.hpp"
// TODO: figure out where to put utils.hpp so this never happens
#include "../../utils.hpp"

// #include <Eigen/Core>

/**
 * The basic list-of-maps sparse matrix.
 */
class DictMatrix : public ISparseMatrix {
  public:
    DictMatrix() = default;
    DictMatrix(long nrows, long ncols) {  // : ncols(ncols), nrows(nrows) {
        this->ncols = ncols;
        this->nrows = nrows;
        // this->matrix = boost::numeric::ublas::coordinate_matrix<long>(this->nrows, this->ncols);
//        this->matrix = std::vector<std::unordered_map<long, long>>(this->nrows, std::unordered_map<long, long>());
        this->matrix = std::vector<MapVector<long>>(this->nrows, MapVector<long>());
        // this->matrix = boost::numeric::ublas::mapped_matrix<long>(this->nrows, this->ncols);
        // long shape_array[2] = {this->nrows, this->ncols};
        this->shape = std::make_pair(this->nrows, this->ncols);
    }
    void add(long row, long col, long val) override;
    // virtual void add(long row, std::vector<long> cols, std::vector<long> values) override;
    void clearrow(long row) override;
    void clearcol(long col) override;
    ISparseMatrix* copy() const override;
    long distinct_edges(long block) const override;
    std::vector<std::tuple<long, long, long>> entries() const override;
    long get(long row, long col) const override;
    std::vector<long> getcol(long col) const override;
    MapVector<long> getcol_sparse(long col) const override;
    const MapVector<long>& getcol_sparseref(long col) const override;
    void getcol_sparse(long col, MapVector<long> &col_vector) const override;
    // virtual MapVector<long> getcol_sparse(long col) override;
    // virtual const MapVector<long>& getcol_sparse(long col) const override;
    std::vector<long> getrow(long row) const override;
    MapVector<long> getrow_sparse(long row) const override;
    void getrow_sparse(long row, MapVector<long> &row_vector) const override;
    const MapVector<long>& getrow_sparseref(long row) const override;
    // virtual MapVector<long> getrow_sparse(long row) override;
    // virtual const MapVector<long>& getrow_sparse(long row) const override;
    EdgeWeights incoming_edges(long block) const override;
    std::set<long> neighbors(long block) const override;
    MapVector<long> neighbors_weights(long block) const override;
    Indices nonzero() const override;
    EdgeWeights outgoing_edges(long block) const override;
    /// Sets the values in a row equal to the input vector
    void setrow(long row, const MapVector<long> &vector) override;
    /// Sets the values in a column equal to the input vector
    void setcol(long col, const MapVector<long> &vector) override;
    void sub(long row, long col, long val) override;
    long edges() const override;
    void print() const override;
    std::vector<long> sum(long axis) const override;
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
    // long ncols;
    // long nrows;
//    std::vector<std::unordered_map<long, long>> matrix;
    std::vector<MapVector<long>> matrix;
    // boost::numeric::ublas::mapped_matrix<long> matrix;
    // boost::numeric::ublas::coordinate_matrix<long> matrix;
};

#endif // CPPSBP_PARTITION_SPARSE_DICT_MATRIX_HPP
