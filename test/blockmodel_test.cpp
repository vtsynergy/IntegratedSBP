#include <vector>

#include <gtest/gtest.h>

#include "blockmodel.hpp"

#include "toy_example.hpp"

class BlockmodelTest : public ToyExample {
protected:
    void SetUp() override {
        args.transpose = true;
        ToyExample::SetUp();
    }
};

class BlockmodelComplexTest : public ComplexExample {
protected:
    void SetUp() override {
        args.transpose = true;
        ComplexExample::SetUp();
    }
};

TEST_F(BlockmodelTest, BlockDegreesAreCorrectlyInstantiated) {
    EXPECT_EQ(B.degrees_out(0), 8);
    EXPECT_EQ(B.degrees_out(1), 7);
    EXPECT_EQ(B.degrees_out(2), 8);
    EXPECT_EQ(B.degrees_in(0), 9);
    EXPECT_EQ(B.degrees_in(1), 7);
    EXPECT_EQ(B.degrees_in(2), 7);
    EXPECT_EQ(B.degrees(0), 10);
    EXPECT_EQ(B.degrees(1), 9);
    EXPECT_EQ(B.degrees(2), 9);
}

TEST_F(BlockmodelTest, MoveVertexWithDenseEdgeCountUpdatesIsCorrect) {
    B.move_vertex(V7, 2, Proposal.proposal, Updates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (long row = 0; row < B.getNum_blocks(); ++row) {
        for (long col = 0; col < B.getNum_blocks(); ++col) {
            long val1 = B.blockmatrix()->get(row, col);
            long val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelTest, MoveVertexWithSparseEdgeCountUpdatesIsCorrect) {
    B.move_vertex(V7, 2, Proposal.proposal, SparseUpdates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (long row = 0; row < B.getNum_blocks(); ++row) {
        for (long col = 0; col < B.getNum_blocks(); ++col) {
            long val1 = B.blockmatrix()->get(row, col);
            long val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelTest, MoveVertexWithBlockmodelDeltasIsCorrect) {
    B.move_vertex(V7, Proposal.proposal, Deltas, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (long row = 0; row < B.getNum_blocks(); ++row) {
        for (long col = 0; col < B.getNum_blocks(); ++col) {
            long val1 = B.blockmatrix()->get(row, col);
            long val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelTest, MoveVertexWithBlockmodelDeltasDynamicBlockDegreesIsCorrect) {
    B.move_vertex(V7, Deltas, Proposal);
    for (long row = 0; row < B.getNum_blocks(); ++row) {
        for (long col = 0; col < B.getNum_blocks(); ++col) {
            long val1 = B.blockmatrix()->get(row, col);
            long val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelTest, MoveVertexWithVertexEdgesIsCorrect) {
    B.move_vertex(Move);
    for (long row = 0; row < B.getNum_blocks(); ++row) {
        for (long col = 0; col < B.getNum_blocks(); ++col) {
            long val1 = B.blockmatrix()->get(row, col);
            long val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelTest, MoveVertexWithSelfEdgesUsingVertexEdgesIsCorrect) {
    std::cout << "Blockmatrix before move: " << std::endl;
    B.print_blockmatrix();
    B.move_vertex(SelfEdgeMove);
    std::cout << "Blockmatrix after move: " << std::endl;
    B.print_blockmatrix();
    std::cout << "Actual blockmatrix: " << std::endl;
    B3.print_blockmatrix();
    for (long row = 0; row < B.getNum_blocks(); ++row) {
        for (long col = 0; col < B.getNum_blocks(); ++col) {
            long val1 = B.blockmatrix()->get(row, col);
            long val2 = B3.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithDenseEdgeCountUpdatesIsCorrect) {
    B.move_vertex(V6, 3, Proposal.proposal, Updates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (long row = 0; row < B.getNum_blocks(); ++row) {
        for (long col = 0; col < B.getNum_blocks(); ++col) {
            long val1 = B.blockmatrix()->get(row, col);
            long val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithSparseEdgeCountUpdatesIsCorrect) {
    B.move_vertex(V6, 3, Proposal.proposal, SparseUpdates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (long row = 0; row < B.getNum_blocks(); ++row) {
        for (long col = 0; col < B.getNum_blocks(); ++col) {
            long val1 = B.blockmatrix()->get(row, col);
            long val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithBlockmodelDeltasIsCorrect) {
    B.move_vertex(V6, Proposal.proposal, Deltas, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (long row = 0; row < B.getNum_blocks(); ++row) {
        for (long col = 0; col < B.getNum_blocks(); ++col) {
            long val1 = B.blockmatrix()->get(row, col);
            long val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithBlockmodelDeltasAndOnTheFlyBlockDegreesIsCorrect) {
    B.move_vertex(V6, Deltas, Proposal);
    for (long row = 0; row < B.getNum_blocks(); ++row) {
        for (long col = 0; col < B.getNum_blocks(); ++col) {
            long val1 = B.blockmatrix()->get(row, col);
            long val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithVertexEdgesIsCorrect) {
    std::cout << "Blockmatrix before move: " << std::endl;
    B.print_blockmatrix();
    B.move_vertex(Move);
    std::cout << "Blockmatrix after move: " << std::endl;
    B.print_blockmatrix();
    std::cout << "Actual blockmatrix: " << std::endl;
    B2.print_blockmatrix();
    for (long row = 0; row < B.getNum_blocks(); ++row) {
        for (long col = 0; col < B.getNum_blocks(); ++col) {
            long val1 = B.blockmatrix()->get(row, col);
            long val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithSelfEdgesUsingVertexEdgesIsCorrect) {
    B.move_vertex(SelfEdgeMove);
    for (long row = 0; row < B.getNum_blocks(); ++row) {
        for (long col = 0; col < B.getNum_blocks(); ++col) {
            long val1 = B.blockmatrix()->get(row, col);
            long val2 = B3.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}
