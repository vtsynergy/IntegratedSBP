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

#include "evaluate.hpp"

class EvaluateTest : public ::testing::Test {
protected:
    // My variables
    std::vector<std::vector<int>> Matrix;
    std::vector<std::vector<int>> Matrix2;
    long N, N2;
    double RandIndex;
    double Precision;
    double Recall;
    double F1, F12;
    void SetUp() override {
        Matrix = {
                { 192, 0,   1,   3,   0,   100, 133, 48 },
                { 0,   232, 0,   0,   0,   1,   0,   0 },
                { 1,   0,   631, 112, 44,  97,  3,   1 },
                { 0,   0,   0,   245, 0,   1,   0,   5 },
                { 0,   0,   0,   0,   172, 75,  151, 6 }
        };
        Matrix2 = {
                {4953, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 4587, 4127, 0, 3854, 0, 0, 1, 1, 0, 4431, 3532, 4, 8, 0, 0, 0, 0, 9, 0, 0, 4, 0, 0, 3183, 3448, 0, 0, 4173, 0, 0, 3303, 3686, 0, 3428, 0, 2, 0, 3545, 3441, 0, 0, 0, 12, 0, 20, 11, 0, 0, 3, 2},
                {0, 2794, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                {0, 0, 1856, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 972, 0, 0, 1, 10, 0, 0, 0, 0, 0, 0, 1647, 1, 1, 0, 0, 903, 606, 1469, 39, 0, 0, 644, 0, 16, 853, 0, 1486, 0, 0, 0, 0, 0, 11, 1487, 0, 0, 0, 0, 0, 968, 3, 0, 0, 1848, 0, 1487, 538, 1688, 338, 569, 0, 1407, 776, 1279},
                {0, 1, 0, 2402, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1986, 1905, 0, 0, 0, 0, 0, 0, 0, 0, 1651, 0, 0, 0, 0, 98, 2, 0, 0, 0, 6, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2302, 0, 7, 0, 8, 28, 0, 0, 11, 0},
                {0, 0, 0, 0, 6147, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 5288, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 5792, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4808, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 8315, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 2905, 0, 0, 0, 0, 0, 0, 0, 2, 2655, 0, 0, 0, 0, 0, 0, 0, 0, 2551, 0, 0, 2293, 0, 0, 2, 6, 0, 1, 0, 0, 2, 2769, 0, 0, 0, 0, 0, 0, 0, 2351, 0, 0, 0, 0, 0, 2695, 0, 2491, 0, 0, 0, 0, 0, 0, 0, 4, 1, 10, 2, 0, 0, 3, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 2539, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2300, 0, 1, 0, 0, 0, 2209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 3, 2173, 0, 2, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4523, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6398, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5704, 1, 0, 0, 0, 0, 0, 0, 0, 5981, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2040, 0, 0, 0, 0, 0, 0, 1563, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 0, 2, 7, 8, 1475, 0, 0, 5, 0, 1454, 0, 0, 1, 0, 0, 0, 0, 0, 1705, 0, 0, 0, 0, 0, 0, 1, 1785, 0, 0, 0, 0, 1, 12, 11, 5, 3, 0, 1, 5, 2},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3199, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3188, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 1, 0},
        };
        RandIndex = 0.841589;
        Recall = 0.508136;
        Precision = 0.797772;
        F1 = 0.620835;
        N = 2254;
        N2 = 200000;
        F12 = 0.26938562;
    }
};

TEST_F(EvaluateTest, CalculateF1ScoreReturnsCorrectF1Score) {
    double f1 = evaluate::calculate_f1_score(N, Matrix);
    EXPECT_FLOAT_EQ(f1, F1)  << "Calculated F1 Score = " << f1 << " but was expecting " << F1;
}

TEST_F(EvaluateTest, CalculateF1ScoreReturnsCorrectF1ScoreOnBiggerMatrix) {
    double f1 = evaluate::calculate_f1_score(N2, Matrix2);
    EXPECT_FLOAT_EQ(f1, F12)  << "Calculated F1 Score = " << f1 << " but was expecting " << F12;
}

