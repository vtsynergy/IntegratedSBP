/// ====================================================================================================================
/// Part of the accelerated Stochastic Block Partitioning (SBP) project.
/// Copyright (C) Virginia Polytechnic Institute and State University, 2023. All Rights Reserved.
///
/// This software is provided as-is. Neither the authors, Virginia Tech nor Virginia Tech Intellectual Properties, Inc.
/// assert, warrant, or guarantee that the software is fit for any purpose whatsoever, nor do they collectively or
/// individually accept any responsibility or liability for any action or activity that results from the use of this
/// software.  The entire risk as to the quality and performance of the software rests with the user, and no remedies
/// shall be provided by the authors, Virginia Tech or Virginia Tech Intellectual Properties, Inc.
/// This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
/// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
/// details.
/// You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to
/// the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.
///
/// Author: Frank Wanye
/// ====================================================================================================================
#include <vector>

#include <gtest/gtest.h>

#include "blockmodel.hpp"
#include "dict_matrix.hpp"

#include "toy_example.hpp"

class DictMatrixTest : public ToyExample {
    void SetUp() override {
        ToyExample::ToySetUp(false);
    }
};

class DictMatrixComplexTest : public ComplexExample {
    void SetUp() override {
        ComplexExample::SetUp();
        ComplexToySetUp(false);
    }
};

TEST_F(DictMatrixTest, NeighborsAreCorrectlyReturned) {
    std::set<long> neighbors = B.blockmatrix()->neighbors(1);
    // neighbors should contain all 3 blocks
    EXPECT_EQ(neighbors.size(), 3);
    for (long i = 0; i < 3; ++i) {
        EXPECT_TRUE(neighbors.find(i) != neighbors.end());
    }
}

TEST_F(DictMatrixTest, UpdateEdgeCountsIsCorrect) {
    B.blockmatrix()->update_edge_counts(Deltas);
    for (long row = 0; row < B.getNum_blocks(); ++row) {
        for (long col = 0; col < B.getNum_blocks(); ++col) {
            long correct_val = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(B.blockmatrix()->get(row, col), correct_val);
        }
    }
}

TEST_F(DictMatrixComplexTest, NeighborsAreCorrectlyReturned) {
    std::set<long> neighbors = B.blockmatrix()->neighbors(1);
    B.print_blockmatrix();
    // neighbors should contain all 3 blocks
    std::vector<long> correct_neighbors = { 0, 3, 4, 5 };
    std::cout << "Correct neighbors: " << std::endl;
    utils::print<long>(correct_neighbors);
    std::cout << "Returned neighbors: " << std::endl;
    std::vector<long> n2(neighbors.begin(), neighbors.end());
    utils::print<long>(n2);
    EXPECT_EQ(neighbors.size(), correct_neighbors.size());
    for (const long neighbor : correct_neighbors) {
        EXPECT_TRUE(neighbors.find(neighbor) != neighbors.end());
    }
}

TEST_F(DictMatrixComplexTest, UpdateEdgeCountsIsCorrect) {
    B.blockmatrix()->update_edge_counts(Deltas);
    for (long row = 0; row < B.getNum_blocks(); ++row) {
        for (long col = 0; col < B.getNum_blocks(); ++col) {
            long correct_val = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(B.blockmatrix()->get(row, col), correct_val);
        }
    }
}

