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
#include "distributed/dist_blockmodel_triplet.hpp"
#include "distributed/two_hop_blockmodel.hpp"

namespace finetune::dist {

/// Stores individual MCMC runtimes per MCMC iteration for checking runtime imbalance
extern std::vector<double> MCMC_RUNTIMES;
extern std::vector<unsigned long> MCMC_VERTEX_EDGES;
extern std::vector<long> MCMC_NUM_BLOCKS;
extern std::vector<unsigned long> MCMC_BLOCK_DEGREES;
extern std::vector<unsigned long long> MCMC_AGGREGATE_BLOCK_DEGREES;

/// Updates `blockmodel` for one membership update contained in `membership`.
void async_move(const Membership &membership, const Graph &graph, TwoHopBlockmodel &blockmodel);

/// Runs the Asynchronous Gibbs algorithm in a distributed fashion using MPI.
TwoHopBlockmodel &asynchronous_gibbs(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels);

/// Runs one iteration of the asynchronous Gibbs algorithm in a distributed fashion using MPI.
std::vector<Membership> asynchronous_gibbs_iteration(TwoHopBlockmodel &blockmodel, const Graph &graph,
                                                     const std::vector<long> &active_set = std::vector<long>(),
                                                     int batch = 0);

/// If the average of the last 3 delta entropies is < threshold * initial_entropy, stop the algorithm.
bool early_stop(long iteration, DistBlockmodelTriplet &blockmodels, double initial_entropy,
                std::vector<double> &delta_entropies);

/// Finetunes the partial results on a given graph.
Blockmodel &finetune_assignment(TwoHopBlockmodel &blockmodel, Graph &graph);

/// Runs the hybrid MCMC algorithm in a distributed fashion using MPI.
TwoHopBlockmodel &hybrid_mcmc(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels);

/// Records metrics that may be causing imbalance.
void measure_imbalance_metrics(const TwoHopBlockmodel &blockmodel, const Graph &graph);

/// Runs the Metropolis Hastings algorithm in a distributed fashion using MPI.
TwoHopBlockmodel &metropolis_hastings(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels);

/// Runs one iteration of the Metropolis-Hastings algorithm. Returns the accepted vertex moves.
std::vector<Membership> metropolis_hastings_iteration(TwoHopBlockmodel &blockmodel, Graph &graph,
                                                      const std::vector<long> &active_set = std::vector<long>(),
                                                      int batch = -1);

/// Proposes an asynchronous Gibbs move in a distributed setting.
VertexMove propose_gibbs_move(const TwoHopBlockmodel &blockmodel, long vertex, const Graph &graph);

/// Proposes a metropolis hastings move in a distributed setting.
VertexMove propose_mh_move(TwoHopBlockmodel &blockmodel, long vertex, const Graph &graph);

}  // namespace finetune::dist