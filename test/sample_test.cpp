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
