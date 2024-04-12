#include <vector>

#include <gtest/gtest.h>

#include "entropy.hpp"
#include "block_merge.hpp"
#include "blockmodel/sparse/delta.hpp"

#include "toy_example.hpp"

TEST_F(BlockMergeTest, BlockmodelDeltaIsCorrectlyComputed) {
    Delta delta = block_merge::blockmodel_delta(0, 1, B);
    EXPECT_EQ(delta.entries().size(), 6);
    EXPECT_EQ(delta.get(0,0), -7);
    EXPECT_EQ(delta.get(0,1), -1);
    EXPECT_EQ(delta.get(1,0), -1);
    EXPECT_EQ(delta.get(1,1), 9);
    EXPECT_EQ(delta.get(1,2), 0);
    EXPECT_EQ(delta.get(2,0), -1);
    EXPECT_EQ(delta.get(2,1), 1);
}
