/// Partition the graph amongst multiple MPI processes
#include <algorithm>
#include <random>
#include <unordered_map>
#include <vector>

#include "args.hpp"
#include "blockmodel/sparse/typedefs.hpp"
#include "graph.hpp"
#include "utils.hpp"

// TODO: If required, create a GraphPartition graph that will contain additional information
namespace partition {

/// Partitions the graph using the method chosen via `args`
Graph partition(const Graph &graph, long rank, long num_processes, Args &args);

/// Partitions the graph using the round robin strategy. The resulting partitions do not overlap.
Graph partition_round_robin(const Graph &graph, long rank, long num_processes, long target_num_vertices);

/// Partitions the graph using the random strategy. The resulting partitions do not overlap.
Graph partition_random(const Graph &graph, long rank, long num_processes, long target_num_vertices);

/// Partitions the graph using the snowball sampling strategy. The resulting partitions DO overlap.
Graph partition_snowball(const Graph &graph, long rank, long num_processes, long target_num_vertices);

}