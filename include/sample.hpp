/**
* Contains code for sampling from graphs.
*/
#ifndef SBP_SAMPLE_HPP
#define SBP_SAMPLE_HPP

#include <vector>

#include "blockmodel/blockmodel.hpp"
#include "graph.hpp"

namespace sample {

struct ES_State {
    ES_State(long num_vertices) {
        this->contribution = utils::constant<long>(num_vertices, 0);
        this->neighborhood_flag = utils::constant<bool>(num_vertices, false);
        this->neighbors = std::set<long>();
        this->contribution_sum = 0;
    }
    std::vector<long> contribution;
    std::vector<bool> neighborhood_flag;
    std::set<long> neighbors;
    long contribution_sum;
};

struct Sample {
    Graph graph;
    std::vector<long> mapping;
};

/// Detaches island and 1-degree vertices from the graph.
Sample detach(const Graph &graph);

/// Samples vertices using edge degree products, which are proportional to fisher information content.
Sample degree_product(const Graph &graph);

/// Samples vertices using edge degree products and snowball sampling, which are proportional to fisher information content.
Sample degree_product_snowball(const Graph &graph);

/// Adds `vertex` to the expansion snowball sample.
void es_add_vertex(const Graph &graph, ES_State &state, std::vector<long> &sampled, std::vector<long> &mapping,
                   long vertex);

/// Updates the contribution of `vertex`, which has just been placed in the neighborhood of the current sample.
/// Also decreases the contribution of all vertices that link to `vertex`.
void es_update_contribution(const Graph &graph, ES_State &state, const std::vector<long> &mapping, long vertex);

/// Samples vertices using the expansion snowball algorithm of Maiya et al.
Sample expansion_snowball(const Graph &graph);

/// Extends the results from the sample graph blockmodel to the full graph blockmodel.
std::vector<long> extend(const Graph &graph, const Blockmodel &sample_blockmodel, const Sample &sample);

/// Creates a Sample from sampled vertices and their mappings.
Sample from_vertices(const Graph &graph, const std::vector<long> &vertices, const std::vector<long> &mapping);

/// Samples vertices with the highest degrees.
Sample max_degree(const Graph &graph);

/// Samples random vertices.
Sample random(const Graph &graph);

/// Samples vertices in round robin fashion, using subgraph_index and num_subgraphs as the round robin parameters.
Sample round_robin(const Graph &graph, int subgraph_index, int num_subgraphs);

/// Samples vertices in a snowball fashion, using subgraph_index and num_subgraphs as the snowball parameters.
Sample snowball(const Graph &graph, int subgraph_index, int num_subgraphs);

/// Creates a sample using args.samplingalg algorithm
Sample sample(const Graph &graph);

/// Re-attaches island and 1-degree vertices to the graph.
Blockmodel reattach(const Graph &graph, const Blockmodel &sample_blockmodel, const Sample &sample);

}

#endif // SBP_SAMPLE_HPP
