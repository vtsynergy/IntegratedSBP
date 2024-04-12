#include "sample.hpp"

#include <chrono>

#include "args.hpp"
#include "common.hpp"
#include "mpi_data.hpp"
#include "random"

namespace sample {

Sample detach(const Graph &graph) {
    std::vector<long> sampled;
    std::vector<long> mapping = utils::constant<long>(graph.num_vertices(), -1);
    std::vector<long> degrees = graph.degrees();
    long index = 0;
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        long degree = degrees[vertex];
        if (degree > 1) {
            sampled.push_back(vertex);
            mapping[vertex] = index;
            index++;
        }
    }
    return from_vertices(graph, sampled, mapping);
}

Sample degree_product(const Graph &graph) {
    std::vector<std::pair<std::pair<long, long>, long>> edge_info = graph.sorted_edge_list();
    std::vector<long> sampled;
    std::vector<long> mapping = utils::constant<long>(graph.num_vertices(), -1);
    int num_sampled = 0;
    for (const std::pair<std::pair<long, long>, long> &edge : edge_info) {
        long source = edge.first.first;
        long destination = edge.first.second;
        if (mapping[source] ==  -1) {
            sampled.push_back(source);
            mapping[source] = num_sampled;
            num_sampled++;
        }
        if (mapping[destination] ==  -1) {
            sampled.push_back(destination);
            mapping[destination] = num_sampled;
            num_sampled++;
        }
        if (num_sampled >= long(args.samplesize * double(graph.num_vertices()))) break;
    }
    return from_vertices(graph, sampled, mapping);
}

Sample degree_product_snowball(const Graph &graph) {
//    std::vector<std::pair<std::pair<long, long>, long>> edge_info = graph.sorted_edge_list();
    std::vector<long> degrees = graph.degrees();
    std::vector<long> sampled;
    std::vector<long> mapping = utils::constant<long>(graph.num_vertices(), -1);
//    std::vector<long> vertex_max_degree_product = utils::constant<long>(graph.num_vertices(), 0);
    typedef std::tuple<long, long, long> edge_t;
    auto cmp_fxn = [](edge_t left, edge_t right) { return std::get<2>(left) < std::get<2>(right); };
    std::priority_queue<edge_t, std::vector<edge_t>, decltype(cmp_fxn)> frontier(cmp_fxn);
    auto dps_add_vertex = [&graph, &frontier, &mapping, &sampled, &degrees](long vertex) {
        mapping[vertex] = (long) sampled.size();
        sampled.push_back(vertex);
        for (const long &neighbor : graph.out_neighbors(vertex)) {
            if (mapping[neighbor] != -1) continue;  // neighbor already sampled
            frontier.push(std::make_tuple(vertex, neighbor, degrees[vertex] * degrees[neighbor]));
        }
        for (const long &neighbor : graph.in_neighbors(vertex)) {
            if (mapping[neighbor] != -1) continue;  // neighbor already sampled
            frontier.push(std::make_tuple(neighbor, vertex, degrees[neighbor] * degrees[vertex]));
        }
    };
    long start = common::random_integer(0, graph.num_vertices() - 1);
    dps_add_vertex(start);
    while (long(sampled.size()) < long(double(graph.num_vertices()) * args.samplesize)) {
        if (frontier.empty()) {  // restart from a new vertex if frontier is empty
            long vertex;
            do {
                vertex = common::random_integer(0, graph.num_vertices() - 1);
            } while (mapping[vertex] >= 0);
            dps_add_vertex(vertex);
        }
        edge_t next = frontier.top();
        frontier.pop();
        long source = std::get<0>(next);
        long destination = std::get<1>(next);
        if (mapping[source] == -1) dps_add_vertex(source);
        if (mapping[destination] == -1) dps_add_vertex(destination);
        // if source and destination both sampled (can happen with directed graphs), neither is added, and we're back in
        // the while loop
    }
    return from_vertices(graph, sampled, mapping);
}

void es_add_vertex(const Graph &graph, ES_State &state, std::vector<long> &sampled, std::vector<long> &mapping,
                   long vertex) {
    sampled.push_back(vertex);
    long index = long(sampled.size()) - 1;
    mapping[vertex] = index;
    for (long neighbor : graph.out_neighbors(vertex)) {
        if (state.neighborhood_flag[neighbor]) continue;  // if already in neighborhood, ignore
        if (mapping[neighbor] >= 0) continue;  // if already sampled neighbor, ignore
        state.neighbors.insert(neighbor);
        state.neighborhood_flag[neighbor] = true;
        if (state.contribution[neighbor] > 0) continue;  // contribution has already been calculated
        es_update_contribution(graph, state, mapping, neighbor);  // this should also set contribution[vertex] to 0
    }
    state.neighbors.erase(vertex);
    state.neighborhood_flag[vertex] = false;
}

void es_update_contribution(const Graph &graph, ES_State &state, const std::vector<long> &mapping, long vertex) {
    for (long neighbor : graph.out_neighbors(vertex)) {
        if (state.neighborhood_flag[neighbor]) continue;
        if (mapping[neighbor] >= 0) continue;
        state.contribution[vertex]++;
        state.contribution_sum++;
    }
    for (long neighbor : graph.in_neighbors(vertex)) {
        if (state.contribution[neighbor] > 0) {
            state.contribution[neighbor]--;
            state.contribution_sum--;
        }
    }
}

Sample expansion_snowball(const Graph &graph) {
    std::vector<long> sampled;
    std::vector<long> mapping = utils::constant<long>(graph.num_vertices(), -1);
    ES_State state(graph.num_vertices());
    long start = common::random_integer(0, graph.num_vertices() - 1);
    es_add_vertex(graph, state, sampled, mapping, start);
    while (long(sampled.size()) < long(double(graph.num_vertices()) * args.samplesize)) {
        if (state.neighbors.empty()) {  // choose random vertex not already sampled
            long vertex;
            // Assuming sample size is < 50% (0.5), this should run less than 2 times on average.
            // If the graph consists of just one connected component, this whole if statement should never run at all.
            do {
                vertex = common::random_integer(0, graph.num_vertices() - 1);
            } while (mapping[vertex] >= 0);
            es_add_vertex(graph, state, sampled, mapping, vertex);
            continue;
        } else if (state.contribution_sum == 0) {  // choose random neighbor
            long index = common::random_integer(0, long(state.neighbors.size()) - 1);
            auto it = state.neighbors.begin();
            std::advance(it, index);
            long vertex = *it;
            es_add_vertex(graph, state, sampled, mapping, vertex);
            continue;
        }
        // choose neighbor with max contribution
        long vertex = utils::argmax<long>(state.contribution);
        es_add_vertex(graph, state, sampled, mapping, vertex);
    }
    return from_vertices(graph, sampled, mapping);
}

std::vector<long> extend(const Graph &graph, const Blockmodel &sample_blockmodel, const Sample &sample) {
    if (mpi.rank == 0) std::cout << "Extending the sample results to the full graph" << std::endl;
    std::vector<long> assignment = utils::constant<long>(graph.num_vertices(), -1);
    // Embed the known assignments from the partitioned sample
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        long sample_vertex = sample.mapping[vertex];
        if (sample_vertex == -1) continue;
        assignment[vertex] = sample_blockmodel.block_assignment(sample_vertex);
    }
    // Infer membership of remaining vertices
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        if (assignment[vertex] != -1) continue;  // already assigned
        // Count edges to/from different communities
        MapVector<long> edge_counts;
        for (long neighbor : graph.out_neighbors(vertex)) {
            long community = assignment[neighbor];
            if (community == -1) continue;  // we don't know neighbor's community
            edge_counts[community]++;
        }
        for (long neighbor : graph.in_neighbors(vertex)) {
            long community = assignment[neighbor];
            if (community == -1 || neighbor == vertex) continue;
            edge_counts[community]++;
        }
        if (edge_counts.empty()) {  // assign random community
            long community = common::random_integer(0, sample_blockmodel.getNum_blocks() - 1);
            assignment[vertex] = community;
            continue;
        }
        long max_edges = 0;
        long likely_community = -1;
        for (const auto &element : edge_counts) {
            long community = element.first;
            long edges = element.second;
            if (edges > max_edges) {
                max_edges = edges;
                likely_community = community;
            }
        }
        assignment[vertex] = likely_community;
    }
    return assignment;
//    return Blockmodel(sample_blockmodel.getNum_blocks(), graph, 0.5, assignment);
}

Sample from_vertices(const Graph &graph, const std::vector<long> &vertices, const std::vector<long> &mapping) {
    Graph sampled_graph(long(vertices.size()));
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        long vertex_id = mapping[vertex];
        if (vertex_id == -1) continue;
        const std::vector<long> &neighbors = graph.out_neighbors(vertex);
        for (long neighbor : neighbors) {
            long neighbor_id = mapping[neighbor];
            if (neighbor_id == -1) continue;
            sampled_graph.add_edge(vertex_id, neighbor_id);
        }
        sampled_graph.assign(vertex_id, graph.assignment(vertex));
    }
    sampled_graph.sort_vertices();
    return Sample { sampled_graph, mapping };
}

Sample max_degree(const Graph &graph) {
    std::vector<long> vertex_degrees = graph.degrees();
    std::vector<long> indices = utils::argsort(vertex_degrees);
//    std::vector<long> indices = utils::range<long>(0, graph.num_vertices());
//    std::sort(indices.data(), indices.data() + indices.size(),  // sort in descending order
//              [vertex_degrees](size_t i1, size_t i2) { return vertex_degrees[i1] > vertex_degrees[i2]; });
    std::vector<long> sampled;
    std::vector<long> mapping = utils::constant<long>(graph.num_vertices(), -1);
    for (long index = 0; index < long(args.samplesize * double(graph.num_vertices())); ++index) {
        long vertex = indices[index];
        sampled.push_back(vertex);
        mapping[vertex] = index;  // from full graph ID to sample graph ID
    }
    return from_vertices(graph, sampled, mapping);
}

Sample random(const Graph &graph) {
    std::vector<long> indices = utils::range<long>(0, graph.num_vertices());
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(indices.begin(), indices.end(), std::mt19937_64(seed));
    std::vector<long> sampled;
    std::vector<long> mapping = utils::constant<long>(graph.num_vertices(), -1);
    for (long index = 0; index < long(args.samplesize * double(graph.num_vertices())); ++index) {
        long vertex = indices[index];
        sampled.push_back(vertex);
        mapping[vertex] = index;  // from full graph ID to sample graph ID
    }
    return from_vertices(graph, sampled, mapping);
}

Sample round_robin(const Graph &graph, int subgraph_index, int num_subgraphs) {
    std::vector<long> sampled;
    std::vector<long> mapping = utils::constant<long>(graph.num_vertices(), -1);
    long index = 0;
    for (long vertex = subgraph_index; vertex < graph.num_vertices(); vertex += num_subgraphs) {
        sampled.push_back(vertex);
	    mapping[vertex] = index;
	    index++;
    }
    return from_vertices(graph, sampled, mapping);
}

Sample snowball(const Graph &graph, int subgraph_index, int num_subgraphs) {
//    MapVector<bool> total_sampled;
    std::vector<int> total_sampled = utils::constant<int>(graph.num_vertices(), -1);
    if (mpi.rank == 0) {  // With LaDiS, this work will be duplicated, but only global rank 0's will be used
        MapVector<bool> unsampled(graph.num_vertices());
        for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
            unsampled[vertex] = true;
        }
        std::vector<MapVector<bool>> sampled(num_subgraphs);
        std::vector<MapVector<bool>> frontiers(num_subgraphs);
        std::vector<long> vertex_degrees = graph.degrees();
        // ============= SETUP ================
        // Start with `num_subgraphs` high degree vertices
        std::vector<int> indices = utils::range<int>(0, graph.num_vertices());
        // I think the problem here is that the starting vertices aren't the same between ranks. Should broadcast the top n
        // vertices out to every rank, or have one rank do the sampling for all ranks and then broadcast results
        std::nth_element(std::execution::par_unseq, indices.data(), indices.data() + num_subgraphs,
                         indices.data() + indices.size(), [&vertex_degrees](size_t i1, size_t i2) {
                    return vertex_degrees[i1] > vertex_degrees[i2];
                });
        // make sure seed vertices are consistent through all subgraphs
    //    std::vector<long> seed_vertices = utils::constant<long>(num_subgraphs, -1);
    //    for (int subgraph = 0; subgraph < num_subgraphs; ++subgraph) {
    //        seed_vertices[subgraph] = indices[subgraph];
    //    }
    //    MPI_Bcast(seed_vertices.data(), (int) seed_vertices.size(), MPI_LONG, 0, mpi.comm);
        // done making sure seed vertices are consistent through all subgraphs
        // Mark vertices as sampled
        for (int subgraph = 0; subgraph < num_subgraphs; ++subgraph) {
            long nth_vertex = indices[subgraph];
            std::cout << mpi.rank << " | " << nth_vertex << " goes to subgraph " << subgraph << std::endl;
            total_sampled[nth_vertex] = subgraph;
            sampled[subgraph][nth_vertex] = true;
            unsampled.erase(nth_vertex);
        }
    //    MPI_Barrier(MPI_COMM_WORLD);
    //    exit(-5);
        // Fill in frontiers
        for (int subgraph = 0; subgraph < num_subgraphs; ++subgraph) {
            std::vector<long> neighbors = graph.neighbors(indices[subgraph]);
            for (const long &vertex : neighbors) {
                if (total_sampled[vertex] != -1) continue;  // vertices that were already sampled shouldn't be in the frontier
                frontiers[subgraph][vertex] = true;
            }
        }
        // ============= END OF SETUP ===============
        // ============= SNOWBALL ==============
        int empty_frontiers = 0;
        while (!unsampled.empty()) {
            for (int subgraph = 0; subgraph < num_subgraphs; ++subgraph) {  // Iterate through the subgraphs
                // TODO: sample highest degree vertex in frontier
                long selected;
                if (frontiers[subgraph].empty()) { // Select a random (first) unsampled vertex.
                    selected = unsampled.begin()->first;
                    empty_frontiers++;
                } else {
                    selected = frontiers[subgraph].begin()->first;
                }
                total_sampled[selected] = subgraph;
                sampled[subgraph][selected] = true;
                unsampled.erase(selected);
                for (MapVector<bool> &frontier : frontiers) {
                    frontier.erase(selected);
                }
                for (const long &neighbor : graph.neighbors(selected)) {
                    if (total_sampled[neighbor] != -1) continue;  // If already sampled, don't add to frontier
                    frontiers[subgraph][neighbor] = true;
                }
            }
        }
        if (mpi.rank == 0) std::cout << "Restarted due to empty frontiers " << empty_frontiers << " times in " << graph.num_vertices() / float(num_subgraphs) << " iterations" << std::endl;
        // ============= END OF SNOWBALL ==============
        std::cout << "Num unsampled: " << unsampled.size() << std::endl;
    }
    // Using MPI_COMM_WORLD explicitly to handle LaDiS. Then the rank is handled by
    MPI_Bcast(total_sampled.data(), (int) total_sampled.size(), MPI_INT, 0, MPI_COMM_WORLD);
    // ============= BOOK-KEEPING ==============
    std::vector<long> sampled_list;
    std::vector<long> mapping = utils::constant<long>(graph.num_vertices(), -1);
    long mapped_index = 0;
    for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        if (total_sampled[vertex] != subgraph_index) continue;  // this vertex goes to another rank
        sampled_list.push_back(vertex);
        mapping[vertex] = mapped_index;
        mapped_index++;
    }
    // ============= END OF BOOK-KEEPING ==============
    return from_vertices(graph, sampled_list, mapping);
}

Sample sample(const Graph &graph) {
    if (args.samplingalg == "max_degree")
        return max_degree(graph);
    else if (args.samplingalg == "random")
        return random(graph);
    else if (args.samplingalg == "expansion_snowball")
        return expansion_snowball(graph);
    else if (args.samplingalg == "degree_product")
        return degree_product(graph);
    else if (args.samplingalg == "degree_product_snowball")
        return degree_product_snowball(graph);
    else
        throw std::invalid_argument(args.samplingalg.append(" is not a valid sampling algorithm!"));
}

Blockmodel reattach(const Graph &graph, const Blockmodel &sample_blockmodel, const Sample &sample) {
    if (mpi.rank == 0) std::cout << "Extending the sample results to the full graph with size: " << graph.num_vertices() << std::endl;
    std::vector<long> assignment = utils::constant<long>(graph.num_vertices(), -1);
    // Embed the known assignments from the partitioned sample
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        long sample_vertex = sample.mapping[vertex];
        if (sample_vertex == -1) continue;
        assignment[vertex] = sample_blockmodel.block_assignment(sample_vertex);
    }
    // Infer membership of remaining vertices
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        if (assignment[vertex] != -1) continue;  // already assigned
        long random_community = common::random_integer(0, sample_blockmodel.getNum_blocks() - 1);
        // Assign to the same community as only neighbor
        for (long neighbor : graph.out_neighbors(vertex)) {
            long community = assignment[neighbor];
            if (community == -1) {
                assignment[vertex] = random_community;
                assignment[neighbor] = random_community;
                break;
            }
            assignment[vertex] = community;
        }
        for (long neighbor : graph.in_neighbors(vertex)) {
            long community = assignment[neighbor];
            if (community == -1) {
                assignment[vertex] = random_community;
                assignment[neighbor] = random_community;
                break;
            }
            assignment[vertex] = community;
        }
        // Vertex is an island
        if (assignment[vertex] < 0) {  // assign random community
            assignment[vertex] = random_community;
        }
    }
    return { sample_blockmodel.getNum_blocks(), graph, 0.5, assignment };
}

}
