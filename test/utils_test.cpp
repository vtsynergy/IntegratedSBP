#include "utils.hpp"

#include "toy_example.hpp"

//MPI_t mpi;  // Unused
//Args args;  // Unused

class UtilsTest : public ToyExample {
public:
    std::vector<std::pair<long, long>> sample_edges;
    std::vector<long> sample_assignment;
    void SetUp() {
        ToyExample::SetUp();
    }
};

TEST_F(UtilsTest, RadixSortBigNumbersIsCorrect) {
    std::vector<long> to_sort = { 0, 10303412354341, 5342, 64545323, 34234324, 43343, 23, 25234 };
    utils::print<long>(to_sort);
    std::vector<long> sorted = { 10303412354341, 64545323, 34234324, 43343, 25234, 5342, 23, 0 };
    utils::radix_sort(to_sort);
    std::cout << "after sorting: ";
    utils::print<long>(to_sort);
    VECTOR_EQ(to_sort, sorted);
    IS_SORTED(to_sort);
}

TEST_F(UtilsTest, RadixSortIsCorrect) {
    std::vector<long> to_sort = { 10, 5, 5, 3, 8, 3, 8, 3, 1, 6, 2, 7, 2, 8, 1 };
    std::vector<long> sorted = { 10, 8, 8, 8, 7, 6, 5, 5, 3, 3, 3, 2, 2, 1, 1 };
    utils::radix_sort(to_sort);
    utils::print<long>(to_sort);
    VECTOR_EQ(to_sort, sorted);
    IS_SORTED(to_sort);
}

TEST_F(UtilsTest, RadixPairSortIsCorrect) {
    std::vector<std::pair<long, long>> to_sort = { {0, 10}, {1, 5}, {2, 5}, {3, 3}, {4, 8}, {5, 3}, {6, 8}, {7, 3}, {8, 1}, {9, 6}, {10, 2}, {11, 7}, {12, 2}, {13, 8}, {14, 1} };
    const auto to_sort_copy = to_sort;
    utils::radix_sort(to_sort);
    for (const auto &p : to_sort) {
        std::cout << "(" << p.first << "," << p.second << "), ";
    }
    std::cout << std::endl;
    for (int i = 0; i < to_sort.size() - 1; ++i) {
        EXPECT_TRUE(to_sort[i].second >= to_sort[i+1].second);
        // Make sure pair.first isn't switched
        int found = false;
        for (const auto &p : to_sort_copy) {
            if (p.first == to_sort[i].first) {
                EXPECT_TRUE(p.second == to_sort[i].second);
                found = true;
            }
        }
        EXPECT_TRUE(found);
    }
    int found = false;
    const auto last_entry = to_sort[to_sort.size() - 1];
    for (const auto &p : to_sort_copy) {
        if (p.first == last_entry.first) {
            EXPECT_TRUE(p.second == last_entry.second);
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(UtilsTest, TemplateSortIsCorrect) {
    std::vector<std::pair<std::pair<long, long>, long>> to_sort = {
            {{0, 10}, 154},
            {{1, 5}, 0},
            {{2, 5}, 1015234213214214213},
            {{3, 3}, 12412515331},
            {{4, 8}, 1},
            {{5, 3}, std::numeric_limits<long>::max()},
            {{6, 8}, 83434231423},
            {{7, 3}, 456234645},
            {{8, 1}, 456234646},
            {{9, 6}, 567},
            {{10, 2}, 632432132},
            {{11, 7}, 1321987873},
            {{12, 2}, 982325112},
            {{13, 8}, 64324216324},
            {{14, 1}, 1} };
    const auto to_sort_copy = to_sort;
    utils::radix_sort<std::pair<long, long>, long>(to_sort);
    for (const auto &p : to_sort) {
        std::cout << "((" << p.first.first << "," << p.first.second << ")," << p.second << "), ";
    }
    std::cout << std::endl;
    for (int i = 0; i < to_sort.size() - 1; ++i) {
        EXPECT_TRUE(to_sort[i].second >= to_sort[i+1].second);
        // Make sure pair.first isn't switched
        int found = false;
        for (const auto &p : to_sort_copy) {
            if (p.first.first == to_sort[i].first.first && p.first.second == to_sort[i].first.second) {
                EXPECT_TRUE(p.second == to_sort[i].second);
                found = true;
            }
        }
        EXPECT_TRUE(found);
    }
    int found = false;
    const auto last_entry = to_sort[to_sort.size() - 1];
    for (const auto &p : to_sort_copy) {
        if (p.first == last_entry.first) {
            EXPECT_TRUE(p.second == last_entry.second);
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(UtilsTest, ArgsortIsCorrect) {
    std::vector<long> to_sort = { 0, 10303412354341, 5342, 64545323, 34234324, 43343, 23, 25234 };
    std::vector<long> indices = utils::argsort(to_sort);
    EXPECT_EQ(indices.size(), to_sort.size());
    EXPECT_EQ(*std::max_element(to_sort.begin(), to_sort.end()), to_sort[indices[0]]);
    EXPECT_EQ(*std::min_element(to_sort.begin(), to_sort.end()), to_sort[indices[indices.size() - 1]]);
    for (int i = 0; i < indices.size() - 1; ++i) {
        EXPECT_TRUE(to_sort[indices[i]] >= to_sort[indices[i + 1]]);
    }
}

//TEST_F(SampleTest, MaxDegreeSamplingIsCorrect) {
//    sample::Sample s = sample::max_degree(graph);
//    EXPECT_EQ(s.graph.num_vertices(), 4);
//    for (long v = 0; v < s.graph.num_vertices(); ++v) {
//        std::cout << "v = " << v << ": ";
//        utils::print<long>(s.graph.out_neighbors(v));
//    }
//    EXPECT_EQ(s.graph.num_edges(), 5);
//    EXPECT_EQ(s.mapping.size(), graph.num_vertices());
//    for (const std::pair<long, long> &edge : sample_edges) {
//        long from = edge.first;
//        long to = edge.second;
//        long found = false;
//        for (long neighbor : s.graph.out_neighbors(from)) {
//            if (neighbor == to) {
//                found = true;
//                continue;
//            }
//        }
//        EXPECT_TRUE(found);
//    }
//    for (long i = 0; i < graph.num_vertices(); ++i) {
//        long sample_vertex = s.mapping[i];
//        if (sample_vertex == -1) continue;
//        EXPECT_EQ(graph.assignment(i), s.graph.assignment(sample_vertex));
//    }
//}
