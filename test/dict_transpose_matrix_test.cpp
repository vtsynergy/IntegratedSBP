#include <vector>

#include <gtest/gtest.h>

#include "blockmodel.hpp"
#include "dict_transpose_matrix.hpp"

#include "toy_example.hpp"

class DictTransposeMatrixTest : public ToyExample {
};

class DictTransposeMatrixComplexTest : public ComplexExample {
};

TEST_F(DictTransposeMatrixTest, NeighborsAreCorrectlyReturned) {
    std::set<long> neighbors = B.blockmatrix()->neighbors(1);
    // neighbors should contain all 3 blocks
    EXPECT_EQ(neighbors.size(), 3);
    for (long i = 0; i < 3; ++i) {
        EXPECT_TRUE(neighbors.find(i) != neighbors.end());
    }
}

TEST_F(DictTransposeMatrixComplexTest, NeighborsAreCorrectlyReturned) {
    std::set<long> neighbors = B.blockmatrix()->neighbors(1);
    // neighbors should contain all 3 blocks
    std::vector<long> correct_neighbors = { 0, 3, 4, 5 };
    EXPECT_EQ(neighbors.size(), correct_neighbors.size());
    for (const long neighbor : correct_neighbors) {
        EXPECT_TRUE(neighbors.find(neighbor) != neighbors.end());
    }
}

