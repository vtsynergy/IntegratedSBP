#include "sample.hpp"

#include "toy_example.hpp"

//MPI_t mpi;  // Unused
//Args args;  // Unused

class SampleTest : public ToyExample {
public:
    std::vector<std::pair<long, long>> sample_edges;
    std::vector<long> sample_assignment;
    void SetUp() override {
        ToyExample::SetUp();
        args.samplesize = 0.4;
        sample_edges = {
                { 0, 0 },
                { 0, 2 },
                { 1, 0 },
                { 1, 3 },
                { 2, 1 },
        };
        assignment = { 1, 0, 2, 0 };
    }
};

TEST_F(SampleTest, MaxDegreeSamplingIsCorrect) {
    sample::Sample s = sample::max_degree(graph);
    EXPECT_EQ(s.graph.num_vertices(), 4);
    for (long v = 0; v < s.graph.num_vertices(); ++v) {
        std::cout << "v = " << v << ": ";
        utils::print<long>(s.graph.out_neighbors(v));
    }
    EXPECT_EQ(s.graph.num_edges(), 5);
    EXPECT_EQ(s.mapping.size(), graph.num_vertices());
    for (const std::pair<long, long> &edge : sample_edges) {
        long from = edge.first;
        long to = edge.second;
        long found = false;
        for (long neighbor : s.graph.out_neighbors(from)) {
            if (neighbor == to) {
                found = true;
                continue;
            }
        }
        EXPECT_TRUE(found);
    }
    for (long i = 0; i < graph.num_vertices(); ++i) {
        long sample_vertex = s.mapping[i];
        if (sample_vertex == -1) continue;
        EXPECT_EQ(graph.assignment(i), s.graph.assignment(sample_vertex));
    }
}
