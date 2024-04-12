/***
 * Sparse Matrix that uses a vector of unordered maps to store the blockmodel.
 */
#ifndef CPPSBP_PARTITION_SPARSE_DIST_DICT_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_DIST_DICT_MATRIX_HPP

#include <mpi.h>
#include <unordered_map>

#include "csparse_matrix.hpp"
#include "mpi_data.hpp"
// TODO: figure out where to put utils.hpp so this never happens
#include "utils.hpp"
#include "typedefs.hpp"

// #include <Eigen/Core>

/**
 * C++ interface of the dictionary (map of maps) sparse matrix
 */
class DistDictMatrix : public IDistSparseMatrix {
  public:
    DistDictMatrix() {}
    // DistDictMatrix(long nrows, long ncols, const MPI &mpi, const std::vector<long> &myblocks) {
    DistDictMatrix(long nrows, long ncols, const std::vector<long> &myblocks) {
        this->ncols = ncols;
        this->nrows = nrows;
        // this->matrix = boost::numeric::ublas::coordinate_matrix<long>(this->nrows, this->ncols);
        this->_matrix = std::vector<std::unordered_map<long, long>>(this->nrows, std::unordered_map<long, long>());
        // this->matrix = boost::numeric::ublas::mapped_matrix<long>(this->nrows, this->ncols);
        // long shape_array[2] = {this->nrows, this->ncols};
        this->shape = std::make_pair(this->nrows, this->ncols);
        // this->sync_ownership(myblocks, mpi);
        this->sync_ownership(myblocks);
    }
    virtual void add(long row, long col, long val) override;
    // virtual void add(long row, std::vector<long> cols, std::vector<long> values) override;
    virtual void clearcol(long col) override;
    virtual void clearrow(long row) override;
    virtual ISparseMatrix* copy() const override;
    virtual IDistSparseMatrix* copyDistSparseMatrix() const override;
    virtual long get(long row, long col) const override;
    virtual std::vector<long> getcol(long col) const override;
    virtual MapVector<long> getcol_sparse(long col) const override;
    virtual void getcol_sparse(long col, MapVector<long> &col_vector) const override;
    // virtual MapVector<long> getcol_sparse(long col) override;
    // virtual const MapVector<long>& getcol_sparse(long col) const override;
    virtual std::vector<long> getrow(long row) const override;
    virtual MapVector<long> getrow_sparse(long row) const override;
    virtual void getrow_sparse(long row, MapVector<long> &row_vector) const override;
    // virtual MapVector<long> getrow_sparse(long row) override;
    // virtual const MapVector<long>& getrow_sparse(long row) const override;
    virtual EdgeWeights incoming_edges(long block) const override;
    virtual Indices nonzero() const override;
    virtual EdgeWeights outgoing_edges(long block) const override;
    // Returns True if this rank owns this block.
    virtual bool stores(long block) const override;
    /// Sets the values in a row equal to the input vector
    virtual void setrow(long row, const MapVector<long> &vector) override;
    /// Sets the values in a column equal to the input vector
    virtual void setcol(long col, const MapVector<long> &vector) override;
    virtual void sub(long row, long col, long val) override;
    virtual long edges() const override;
    virtual std::vector<long> sum(long axis = 0) const override;
    virtual long trace() const override;
    virtual void update_edge_counts(long current_block, long proposed_block, std::vector<long> current_row,
                                    std::vector<long> proposed_row, std::vector<long> current_col,
                                    std::vector<long> proposed_col) override;
    void update_edge_counts(const PairIndexVector &delta) override;
    virtual std::vector<long> values() const override;

  private:
    std::vector<std::unordered_map<long, long>> _matrix;
    // std::vector<long> _ownership;
    /// Syncs the ownership between all MPI processes.
    // void sync_ownership(const std::vector<long> &myblocks, const MPI &mpi) {
    virtual void sync_ownership(const std::vector<long> &myblocks) override;
    // void sync_ownership(const std::vector<long> &myblocks) {
    //     long numblocks[mpi.num_processes];
    //     long num_blocks = myblocks.size();
    //     MPI_Allgather(&(num_blocks), 1, MPI_long, &numblocks, 1, MPI_long, mpi.comm);
    //     long offsets[mpi.num_processes];
    //     offsets[0] = 0;
    //     for (long i = 1; i < mpi.num_processes; ++i) {
    //         offsets[i] = offsets[i-1] + numblocks[i-1];
    //     }
    //     long global_num_blocks = offsets[mpi.num_processes-1] + numblocks[mpi.num_processes-1];
    //     this->_ownership = std::vector<long>(global_num_blocks, -1);
    //     std::cout << "rank: " << mpi.rank << " num_blocks: " << num_blocks << " and globally: " << global_num_blocks << std::endl;
    //     std::vector<long> allblocks(global_num_blocks, -1);
    //     MPI_Allgatherv(myblocks.data(), num_blocks, MPI_long, allblocks.data(), &(numblocks[0]), &(offsets[0]), MPI_long, mpi.comm);
    //     if (mpi.rank == 0) {
    //         utils::print<long>(allblocks);
    //     }
    //     long owner = 0;
    //     for (long i = 0; i < global_num_blocks; ++i) {
    //         if (owner < mpi.num_processes - 1 && i >= offsets[owner+1]) {
    //             owner++;
    //         }
    //         this->_ownership[allblocks[i]] = owner;
    //     }
    //     if (mpi.rank == 0) {
    //         utils::print<long>(this->_ownership);
    //     }
    // }
};

#endif // CPPSBP_PARTITION_SPARSE_DIST_DICT_MATRIX_HPP
