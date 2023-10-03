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