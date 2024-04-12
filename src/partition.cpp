#include "partition.hpp"

Graph partition::partition(const Graph &graph, long rank, long num_processes, Args &args) {
    long target_num_vertices = graph.num_vertices() / num_processes;
    std::cout << "target num vertices = " << target_num_vertices << std::endl;
    if (target_num_vertices == graph.num_vertices())
        return graph;
    if (args.subgraphpartition == "round_robin")
        return partition_round_robin(graph, rank, num_processes, target_num_vertices);
    if (args.subgraphpartition == "random")
        return partition_random(graph, rank, num_processes, target_num_vertices);
    if (args.subgraphpartition == "snowball")
        return partition_snowball(graph, rank, num_processes, target_num_vertices);
    std::cout << "The partition method " << args.subgraphpartition << " doesn't exist. Defaulting to round robin." << std::endl;
    return partition_round_robin(graph, rank, num_processes, target_num_vertices);
}

Graph partition::partition_round_robin(const Graph &graph, long rank, long num_processes, long target_num_vertices) {
    NeighborList in_neighbors(target_num_vertices);
    NeighborList out_neighbors(target_num_vertices);
    long num_vertices = 0, num_edges = 0;
    std::unordered_map<long, long> translator;
    std::vector<bool> self_edges;
    for (long i = rank; i < (long) graph.out_neighbors().size(); i += num_processes) {
        if (utils::insert(translator, i, num_vertices))
            num_vertices++;
        long from = translator[i];  // TODO: can avoid additional lookups by returning the inserted element in insert
        for (long neighbor : graph.out_neighbors(i)) {
            if ((neighbor % num_processes) - rank == 0) {
                if (utils::insert(translator, neighbor, num_vertices))
                    num_vertices++;
                long to = translator[neighbor];
                utils::insert(out_neighbors, from, to);
                utils::insert(in_neighbors, to, from);
                num_edges++;
                while (self_edges.size() < num_vertices) {
                    self_edges.push_back(false);
                }
                if (from == to) {
                    self_edges[from] = true;
                }
            }
        }
    }
    std::vector<long> assignment(num_vertices, -1);
    for (const std::pair<const long, long> &element : translator) {
        assignment[element.second] = graph.assignment(element.first);
    }
    std::cout << "NOTE: rank " << rank << "/" << num_processes - 1 << " has N = " << num_vertices << " E = ";
    std::cout << num_edges << std::endl;
    return Graph(out_neighbors, in_neighbors, num_vertices, num_edges, self_edges, assignment);
}

Graph partition::partition_random(const Graph &graph, long rank, long num_processes, long target_num_vertices) {
    NeighborList in_neighbors(target_num_vertices);
    NeighborList out_neighbors(target_num_vertices);
    long num_vertices = 0, num_edges = 0;
    std::unordered_map<long, long> translator;
    std::vector<bool> self_edges;
    long seed = 1234;  // TODO: make this a command-line argument
    std::vector<long> vertices = utils::range<long>(0, graph.num_vertices());
    std::shuffle(vertices.begin(), vertices.end(), std::default_random_engine(seed));
    std::vector<bool> sampled(graph.num_vertices(), false);
    for (long i = 0; i < target_num_vertices; ++i) {
        long index = (rank * target_num_vertices) + i;
        if (index >= graph.num_vertices()) break;
        sampled[vertices[index]] = true;
        translator[vertices[index]] = num_vertices;
        num_vertices++;
    }
    for (long i = 0; i < (long) graph.out_neighbors().size(); ++i) {
        if (!sampled[i]) continue;
        long from = translator[i];
        for (long neighbor : graph.out_neighbors(i)) {
            if (!sampled[neighbor]) continue;
            long to = translator[neighbor];
            utils::insert(out_neighbors, from, to);
            utils::insert(in_neighbors, to, from);
            num_edges++;
            while (self_edges.size() < num_vertices) {
                self_edges.push_back(false);
            }
            if (from == to) {
                self_edges[from] = true;
            }
        }
    }
    std::vector<long> assignment(num_vertices, -1);
    for (const std::pair<const long, long> &element : translator) {
        assignment[element.second] = graph.assignment(element.first);
    }
    std::cout << "NOTE: rank " << rank << "/" << num_processes - 1 << " has N = " << num_vertices << " E = ";
    std::cout << num_edges << std::endl;
    return Graph(out_neighbors, in_neighbors, num_vertices, num_edges, self_edges, assignment);
}

Graph partition::partition_snowball(const Graph &graph, long rank, long num_processes, long target_num_vertices) {
    NeighborList in_neighbors(target_num_vertices);
    NeighborList out_neighbors(target_num_vertices);
    long num_vertices = 0, num_edges = 0;
    std::unordered_map<long, long> translator;
    std::vector<bool> self_edges;
    // Set up random number generator
    std::default_random_engine generator;
    std::uniform_int_distribution<long> distribution(0, graph.num_vertices() - 1);
    std::vector<bool> sampled(graph.num_vertices(), false);
    std::vector<bool> neighborhood(graph.num_vertices(), false);
    std::vector<long> neighbors;
    std::vector<long> new_neighbors;
    long start;
    while (num_vertices < target_num_vertices) {
        if (neighbors.size() == 0) {  // Start/restart from a new random location
            start = distribution(generator);
            // TODO: replace this with a weighted distribution where sampled vertices have a weight of 0
            while (sampled[start]) {  // keep sampling until you find an unsampled vertex
                start = distribution(generator);
            }
            sampled[start] = true;
            neighborhood[start] = false;  // this is just a precaution, shouldn't need to be set
            translator[start] = num_vertices;
            num_vertices++;
            for (long neighbor : graph.out_neighbors(start)) {
                if (!sampled[neighbor] && !neighborhood[neighbor]) {
                    neighborhood[neighbor] = true;
                    neighbors.push_back(neighbor);
                }
            }
        }
        for (long neighbor : neighbors) {  // snowball from the current list of neighbors
            if (num_vertices == target_num_vertices) break;
            if (!sampled[neighbor]) {
                sampled[neighbor] = true;
                neighborhood[neighbor] = false;
                translator[neighbor] = num_vertices;
                num_vertices++;
                for (long new_neighbor : graph.out_neighbors(neighbor)) {
                    if (!sampled[new_neighbor] && !neighborhood[new_neighbor]) {
                        neighborhood[new_neighbor] = true;
                        new_neighbors.push_back(new_neighbor);
                    }
                }
            }
        }
        neighbors = std::vector<long>(new_neighbors);
        new_neighbors = std::vector<long>();
    }
    for (long i = 0; i < (long) graph.out_neighbors().size(); ++i) {
        if (!sampled[i]) continue;
        long from = translator[i];
        for (long neighbor : graph.out_neighbors(i)) {
            if (!sampled[neighbor]) continue;
            long to = translator[neighbor];
            utils::insert(out_neighbors, from, to);
            utils::insert(in_neighbors, to, from);
            num_edges++;
            while (self_edges.size() < num_vertices) {
                self_edges.push_back(false);
            }
            if (from == to) {
                self_edges[from] = true;
            }
        }
    }
    std::vector<long> assignment(num_vertices, -1);
    for (const std::pair<const long, long> &element : translator) {
        assignment[element.second] = graph.assignment(element.first);
    }
    std::cout << "NOTE: rank " << rank << "/" << num_processes - 1 << " has N = " << num_vertices << " E = ";
    std::cout << num_edges << std::endl;
    return Graph(out_neighbors, in_neighbors, num_vertices, num_edges, self_edges, assignment);
}