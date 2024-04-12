/***
 * A global structure holding MPI information.
 */
#ifndef SBP_MPI_DATA_HPP
#define SBP_MPI_DATA_HPP

#include "mpi.h"
#include <numeric>

// The message types - offset by 100,000,000 because we do not expect more than 1 billion nodes. int32 can hold up to
// +2.147 billion, and we're leaving space for up to 21 message types
const int MSG_GETROW = 1E8;
const int MSG_GETCOL = 2E8;
const int MSG_SENDROW = 3E8;
const int MSG_SENDCOL = 4E8;
const int MSG_SIZEROW = 5E8;
const int MSG_SIZECOL = 6E8;

struct MPI_t {
    int rank;           // The rank of the current process
    int num_processes;  // The total number of processes
    MPI_Comm comm = MPI_COMM_WORLD;      // Communicator to use
};

extern MPI_t mpi;

#endif  // SBP_MPI_DATA_HPP