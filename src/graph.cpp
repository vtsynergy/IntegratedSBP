#include "graph.hpp"

#include <execution>
#include "mpi.h"

#include "utils.hpp"
#include "mpi_data.hpp"

void Graph::add_edge(long from, long to) {
    utils::insert_nodup(this->_out_neighbors, from , to);
    utils::insert_nodup(this->_in_neighbors, to, from);
    this->_num_edges++;
    if (from == to) {
        this->_self_edges[from] = true;
    }
    // TODO: undirected version?
}

std::vector<long> Graph::degrees() const {
    std::vector<long> vertex_degrees;
    for (long vertex = 0; vertex < this->_num_vertices; ++vertex) {
        vertex_degrees.push_back(long(this->_out_neighbors[vertex].size() + this->_in_neighbors[vertex].size()
                                 - this->_self_edges[vertex]));
    }
    return vertex_degrees;
}

Graph Graph::load() {
    // TODO: Add capability to process multiple "streaming" graph parts
    std::string base_path = utils::build_filepath();
    fs::path graph_path = base_path + ".tsv";
    fs::path truth_path = base_path + "_truePartition.tsv";
    // TODO: Handle weighted graphs
    std::vector<std::vector<std::string>> csv_contents = utils::read_csv(graph_path);
    if (csv_contents.empty()) {
        graph_path = base_path + ".mtx";
        csv_contents = utils::read_csv(graph_path);
    }
    Graph graph;
    if (csv_contents[0][0] == "%%MatrixMarket") {
        graph = Graph::load_matrix_market(csv_contents);
    } else {
        graph = Graph::load_text(csv_contents);
    }
    if (mpi.rank == 0)
        std::cout << "V: " << graph.num_vertices() << " E: " << graph.num_edges() << std::endl;

    csv_contents = utils::read_csv(truth_path);
    std::vector<long> assignment;
    // TODO: vertices, communities should be size_t or ulong. Will need to make sure -1 returns are properly handled
    // elsewhere.
    if (!csv_contents.empty()) {
        for (std::vector<std::string> &assign: csv_contents) {
            long vertex = std::stoi(assign[0]) - 1;
            long community = std::stoi(assign[1]) - 1;
            if (vertex >= (long)assignment.size()) {
                std::vector<long> padding(vertex - assignment.size() + 1, -1);
                assignment.insert(assignment.end(), padding.begin(), padding.end());
            }
            assignment[vertex] = community;
        }
    } else {
        assignment = utils::constant<long>(graph.num_vertices(), 0);
    }
    graph.assignment(assignment);
    return graph;
}

/// Loads the graph if it's in a matrix market format.
Graph Graph::load_matrix_market(std::vector<std::vector<std::string>> &csv_contents) {
    if (csv_contents[0][2] != "coordinate") {
        std::cerr << "ERROR " << "Dense matrices are not supported!" << std::endl;
        exit(-1);
    }
    if (csv_contents[0][4] == "symmetric") {
        std::cout << "Graph is symmetric" << std::endl;
        args.undirected = true;
    }
    // Find index at which edges start
    long index = 0;
    long num_vertices, num_edges;
    for (long i = 0; i < csv_contents.size(); ++i) {
        const std::vector<std::string> &line = csv_contents[i];
//        std::cout << "line: ";
//        utils::print<std::string>(line);
        if (line[0][0] == '%') continue;
        num_vertices = std::stoi(line[0]);
        if (num_vertices != std::stoi(line[1])) {
            std::cerr << "ERROR " << "Rectangular matrices are not supported!" << std::endl;
            exit(-1);
        }
        num_edges = std::stoi(line[2]);
        index = i + 1;
        break;
    }
    NeighborList out_neighbors;
    NeighborList in_neighbors;
    std::vector<bool> self_edges = utils::constant<bool>(num_vertices, false);
    for (long i = index; i < csv_contents.size(); ++i) {
        const std::vector<std::string> &edge = csv_contents[i];
        long from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        long to = std::stoi(edge[1]) - 1;
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        utils::insert_nodup(out_neighbors, from , to);
        utils::insert_nodup(in_neighbors, to , from);
        if (args.undirected && from != to) {  // Force symmetric graph to be directed by including reverse edges.
            utils::insert_nodup(out_neighbors, to, from);
            utils::insert_nodup(in_neighbors, from , to);
            num_edges++;
        }
        if (from == to) {
            self_edges[from] = true;
        }
    }
    // Pad the neighbors lists
    while (out_neighbors.size() < size_t(num_vertices)) {
        out_neighbors.push_back(std::vector<long>());
    }
    while (in_neighbors.size() < size_t(num_vertices)) {
        in_neighbors.push_back(std::vector<long>());
    }
    return Graph(out_neighbors, in_neighbors, num_vertices, num_edges, self_edges);
}

/// Loads the graph if it's in a text format: a list of "from to" string pairs.
Graph Graph::load_text(std::vector<std::vector<std::string>> &csv_contents) {
    NeighborList out_neighbors;
    NeighborList in_neighbors;
    std::vector<bool> self_edges;
    long num_vertices = 0;
    if (args.undirected)
        Graph::parse_undirected(in_neighbors, out_neighbors, num_vertices, self_edges, csv_contents);
    else
        Graph::parse_directed(in_neighbors, out_neighbors, num_vertices, self_edges, csv_contents);
    long num_edges = 0;  // TODO: unnecessary re-counting of edges?
    for (const std::vector<long> &neighborhood : out_neighbors) {
        num_edges += (long)neighborhood.size();
    }
    if (args.undirected) {
        num_edges /= 2;
    }
    return Graph(out_neighbors, in_neighbors, num_vertices, num_edges, self_edges);
}

double Graph::modularity(const std::vector<long> &assignment) const {
    // See equation for Q_d in: https://hal.archives-ouvertes.fr/hal-01231784/document
    double result = 0.0;
    for (long vertex_i = 0; vertex_i < this->_num_vertices; ++vertex_i) {
        for (long vertex_j = 0; vertex_j < this->_num_vertices; ++vertex_j) {
            if (assignment[vertex_i] != assignment[vertex_j]) continue;
            long edge_weight = 0.0;
            for (long neighbor : this->_out_neighbors[vertex_i]) {
                if (neighbor == vertex_j) {
                    edge_weight = 1.0;
                    break;
                }
            }
            long deg_out_i = long(this->_out_neighbors[vertex_i].size());
            long deg_in_j = long(this->_in_neighbors[vertex_j].size());
            double temp = edge_weight - (double(deg_out_i * deg_in_j) / double(this->_num_edges));
            result += temp;
        }
    }
    result /= double(this->_num_edges);
    return result;
}

std::vector<long> Graph::neighbors(long vertex) const {
    std::vector<long> all_neighbors;
    for (const long &out_neighbor : this->_out_neighbors[vertex]) {
        all_neighbors.push_back(out_neighbor);
    }
    for (const long &in_neighbor : this->_in_neighbors[vertex]) {
        all_neighbors.push_back(in_neighbor);
    }
    return all_neighbors;
}

void Graph::parse_directed(NeighborList &in_neighbors, NeighborList &out_neighbors, long &num_vertices,
                           std::vector<bool> &self_edges, std::vector<std::vector<std::string>> &contents) {
    for (std::vector<std::string> &edge : contents) {
        long from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        long to = std::stoi(edge[1]) - 1;
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        utils::insert_nodup(out_neighbors, from , to);
        utils::insert_nodup(in_neighbors, to, from);
        while (self_edges.size() < num_vertices) {
            self_edges.push_back(false);
        }
        if (from == to) {
            self_edges[from] = true;
        }
    }
    while (out_neighbors.size() < size_t(num_vertices)) {
        out_neighbors.push_back(std::vector<long>());
    }
    while (in_neighbors.size() < size_t(num_vertices)) {
        in_neighbors.push_back(std::vector<long>());
    }
}

void Graph::parse_undirected(NeighborList &in_neighbors, NeighborList &out_neighbors, long &num_vertices,
                             std::vector<bool> &self_edges, std::vector<std::vector<std::string>> &contents) {
    for (std::vector<std::string> &edge : contents) {
        long from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        long to = std::stoi(edge[1]) - 1;
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        utils::insert_nodup(out_neighbors, from , to);
        if (from != to)
            utils::insert_nodup(out_neighbors, to, from);
        while (self_edges.size() < num_vertices) {
            self_edges.push_back(false);
        }
        if (from == to) {
            self_edges[from] = true;
        }
    }
    in_neighbors = NeighborList(out_neighbors);
    while (out_neighbors.size() < size_t(num_vertices)) {
        out_neighbors.push_back(std::vector<long>());
    }
    while (in_neighbors.size() < size_t(num_vertices)) {
        in_neighbors.push_back(std::vector<long>());
    }
}

void Graph::sort_vertices() {
    if (args.degreeproductsort) {
        this->degree_product_sort();
        return;
    }
//    std::cout << "Starting to sort vertices" << std::endl;
//    double start_t = MPI_Wtime();
    std::vector<long> vertex_degrees = this->degrees();
    std::vector<int> indices = utils::range<int>(0, this->_num_vertices);
    std::nth_element(std::execution::par_unseq, indices.data(), indices.data() + int(args.mh_percent * this->_num_vertices),
              indices.data() + indices.size(), [&vertex_degrees](size_t i1, size_t i2) {
              return vertex_degrees[i1] > vertex_degrees[i2];
    });
    // std::sort(std::execution::par_unseq, indices.data(), indices.data() + indices.size(),  // sort in descending order
    //           [vertex_degrees](size_t i1, size_t i2) { return vertex_degrees[i1] > vertex_degrees[i2]; });
    for (int index = 0; index < this->_num_vertices; ++index) {
        int vertex = indices[index];
        if (index < (args.mh_percent * this->_num_vertices)) {
//            std::cout << "high degree vertex: " << vertex << " degree = " << vertex_degrees[vertex] << std::endl;
            this->_high_degree_vertices.push_back(vertex);
        } else {
//            std::cout << "low degree vertex: " << vertex << " degree = " << vertex_degrees[vertex] << std::endl;
            this->_low_degree_vertices.push_back(vertex);
        }
    }
//    std::cout << "Done sorting vertices, time = " << MPI_Wtime() - start_t << "s" << std::endl;
//    std::cout << "Range = " << *std::min_element(vertex_degrees.begin(), vertex_degrees.end()) << " - " << *std::max_element(vertex_degrees.begin(), vertex_degrees.end()) << std::endl;
    int num_islands = 0;
    for (int deg : vertex_degrees) {
        if (deg == 0) num_islands++;
    }
    std::cout << "Num island vertices = " << num_islands << std::endl;
}

void Graph::degree_product_sort() {
//    std::cout << "Starting to sort vertices based on influence" << std::endl;
//    double start_t = MPI_Wtime();
    std::vector<std::pair<std::pair<long, long>, long>> edge_info = this->sorted_edge_list();
    MapVector<bool> selected;
    int num_to_select = int(args.mh_percent * this->_num_vertices);
    int edge_index = 0;
    while (selected.size() < num_to_select) {
        const std::pair<std::pair<long, long>, long> &edge = edge_info[edge_index];
        selected[edge.first.first] = true;
        selected[edge.first.second] = true;
        edge_index++;
    }
    for (const std::pair<long, bool> &entry : selected) {
        this->_high_degree_vertices.push_back(entry.first);
    }
    for (long vertex = 0; vertex < this->_num_vertices; ++vertex) {
        if (selected[vertex]) continue;
        this->_low_degree_vertices.push_back(vertex);
    }
//    std::cout << "Done sorting vertices, time = " << MPI_Wtime() - start_t << "s" << std::endl;
}

long Graph::num_islands() const {
    std::vector<long> vertex_degrees = this->degrees();
    long num_islands = 0;
    for (const long &degree : vertex_degrees) {
        if (degree == 0) num_islands++;
    }
    return num_islands;
}

std::vector<std::pair<std::pair<long, long>, long>> Graph::sorted_edge_list() const {
    std::vector<long> vertex_degrees = this->degrees();
    std::vector<std::pair<std::pair<long, long>, long>> edge_info;
    for (long source = 0; source < this->_num_vertices; ++source) {
        const std::vector<long> &neighbors = this->_out_neighbors[source];
        for (const long &dest : neighbors) {
            long information = vertex_degrees[source] * vertex_degrees[dest];
            edge_info.emplace_back(std::make_pair(source, dest), information);
        }
    }
    std::sort(std::execution::par_unseq, edge_info.begin(), edge_info.end(), [](const auto &i1, const auto &i2) {
        return i1.second > i2.second;
    });
    return edge_info;
}
