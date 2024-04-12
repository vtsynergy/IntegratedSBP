// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/random/log_uniform_int_distribution.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/random/internal/chi_square.h"
#include "absl/random/internal/distribution_test_util.h"
#include "absl/random/internal/pcg_engine.h"
#include "absl/random/internal/sequence_urbg.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/strip.h"

namespace {

template <typename IntType>
class LogUniformIntDistributionTypeTest : public ::testing::Test {};

using IntTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t,  //
                                  uint8_t, uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(LogUniformIntDistributionTypeTest, IntTypes);

TYPED_TEST(LogUniformIntDistributionTypeTest, SerializeTest) {
  using param_type =
      typename absl::log_uniform_int_distribution<TypeParam>::param_type;
  using Limits = std::numeric_limits<TypeParam>;

  constexpr int kCount = 1000;
  absl::InsecureBitGen gen;
  for (const auto& param : {
           param_type(0, 1),                             //
           param_type(0, 2),                             //
           param_type(0, 2, 10),                         //
           param_type(9, 32, 4),                         //
           param_type(1, 101, 10),                       //
           param_type(1, Limits::max() / 2),             //
           param_type(0, Limits::max() - 1),             //
           param_type(0, Limits::max(), 2),              //
           param_type(0, Limits::max(), 10),             //
           param_type(Limits::min(), 0),                 //
           param_type(Limits::lowest(), Limits::max()),  //
           param_type(Limits::min(), Limits::max()),     //
       }) {
    // Validate parameters.
    const auto min = param.min();
    const auto max = param.max();
    const auto base = param.base();
    absl::log_uniform_int_distribution<TypeParam> before(min, max, base);
    EXPECT_EQ(before.min(), param.min());
    EXPECT_EQ(before.max(), param.max());
    EXPECT_EQ(before.base(), param.base());

    {
      absl::log_uniform_int_distribution<TypeParam> via_param(param);
      EXPECT_EQ(via_param, before);
    }

    // Validate stream serialization.
    std::stringstream ss;
    ss << before;

    absl::log_uniform_int_distribution<TypeParam> after(3, 6, 17);

    EXPECT_NE(before.max(), after.max());
    EXPECT_NE(before.base(), after.base());
    EXPECT_NE(before.param(), after.param());
    EXPECT_NE(before, after);

    ss >> after;

    EXPECT_EQ(before.min(), after.min());
    EXPECT_EQ(before.max(), after.max());
    EXPECT_EQ(before.base(), after.base());
    EXPECT_EQ(before.param(), after.param());
    EXPECT_EQ(before, after);

    // Smoke test.
    auto sample_min = after.max();
    auto sample_max = after.min();
    for (int i = 0; i < kCount; i++) {
      auto sample = after(gen);
      EXPECT_GE(sample, after.min());
      EXPECT_LE(sample, after.max());
      if (sample > sample_max) sample_max = sample;
      if (sample < sample_min) sample_min = sample;
    }
    LOG(INFO) << "Range: " << sample_min << ", " << sample_max;
  }
}

using log_uniform_i32 = absl::log_uniform_int_distribution<int32_t>;

class LogUniformIntChiSquaredTest
    : public testing::TestWithParam<log_uniform_i32::param_type> {
 public:
  // The ChiSquaredTestImpl provides a chi-squared goodness of fit test for
  // data generated by the log-uniform-int distribution.
  double ChiSquaredTestImpl();

  // We use a fixed bit generator for distribution accuracy tests.  This allows
  // these tests to be deterministic, while still testing the qualify of the
  // implementation.
  absl::random_internal::pcg64_2018_engine rng_{0x2B7E151628AED2A6};
};

double LogUniformIntChiSquaredTest::ChiSquaredTestImpl() {
  using absl::random_internal::kChiSquared;

  const auto& param = GetParam();

  // Check the distribution of L=log(log_uniform_int_distribution, base),
  // expecting that L is roughly uniformly distributed, that is:
  //
  //   P[L=0] ~= P[L=1] ~= ... ~= P[L=log(max)]
  //
  // For a total of X entries, each bucket should contain some number of samples
  // in the interval [X/k - a, X/k + a].
  //
  // Where `a` is approximately sqrt(X/k). This is validated by bucketing
  // according to the log function and using a chi-squared test for uniformity.

  const bool is_2 = (param.base() == 2);
  const double base_log = 1.0 / std::log(param.base());
  const auto bucket_index = [base_log, is_2, &param](int32_t x) {
    uint64_t y = static_cast<uint64_t>(x) - param.min();
    return (y == 0) ? 0
                    : is_2 ? static_cast<int>(1 + std::log2(y))
                           : static_cast<int>(1 + std::log(y) * base_log);
  };
  const int max_bucket = bucket_index(param.max());  // inclusive
  const size_t trials = 15 + (max_bucket + 1) * 10;

  log_uniform_i32 dist(param);

  std::vector<int64_t> buckets(max_bucket + 1);
  for (size_t i = 0; i < trials; ++i) {
    const auto sample = dist(rng_);
    // Check the bounds.
    ABSL_ASSERT(sample <= dist.max());
    ABSL_ASSERT(sample >= dist.min());
    // Convert the output of the generator to one of num_bucket buckets.
    int bucket = bucket_index(sample);
    ABSL_ASSERT(bucket <= max_bucket);
    ++buckets[bucket];
  }

  // The null-hypothesis is that the distribution is uniform with respect to
  // log-uniform-int bucketization.
  const int dof = buckets.size() - 1;
  const double expected = trials / static_cast<double>(buckets.size());

  const double threshold = absl::random_internal::ChiSquareValue(dof, 0.98);

  double chi_square = absl::random_internal::ChiSquareWithExpected(
      std::begin(buckets), std::end(buckets), expected);

  const double p = absl::random_internal::ChiSquarePValue(chi_square, dof);

  if (chi_square > threshold) {
    LOG(INFO) << "values";
    for (size_t i = 0; i < buckets.size(); i++) {
      LOG(INFO) << i << ": " << buckets[i];
    }
    LOG(INFO) << "trials=" << trials << "\n"
              << kChiSquared << "(data, " << dof << ") = " << chi_square << " ("
              << p << ")\n"
              << kChiSquared << " @ 0.98 = " << threshold;
  }
  return p;
}

TEST_P(LogUniformIntChiSquaredTest, MultiTest) {
  const int kTrials = 5;
  int failures = 0;
  for (int i = 0; i < kTrials; i++) {
    double p_value = ChiSquaredTestImpl();
    if (p_value < 0.005) {
      failures++;
    }
  }

  // There is a 0.10% chance of producing at least one failure, so raise the
  // failure threshold high enough to allow for a flake rate < 10,000.
  EXPECT_LE(failures, 4);
}

// Generate the parameters for the test.
std::vector<log_uniform_i32::param_type> GenParams() {
  using Param = log_uniform_i32::param_type;
  using Limits = std::numeric_limits<int32_t>;

  return std::vector<Param>{
      Param{0, 1, 2},
      Param{1, 1, 2},
      Param{0, 2, 2},
      Param{0, 3, 2},
      Param{0, 4, 2},
      Param{0, 9, 10},
      Param{0, 10, 10},
      Param{0, 11, 10},
      Param{1, 10, 10},
      Param{0, (1 << 8) - 1, 2},
      Param{0, (1 << 8), 2},
      Param{0, (1 << 30) - 1, 2},
      Param{-1000, 1000, 10},
      Param{0, Limits::max(), 2},
      Param{0, Limits::max(), 3},
      Param{0, Limits::max(), 10},
      Param{Limits::min(), 0},
      Param{Limits::min(), Limits::max(), 2},
  };
}

std::string ParamName(
    const ::testing::TestParamInfo<log_uniform_i32::param_type>& info) {
  const auto& p = info.param;
  std::string name =
      absl::StrCat("min_", p.min(), "__max_", p.max(), "__base_", p.base());
  return absl::StrReplaceAll(name, {{"+", "_"}, {"-", "_"}, {".", "_"}});
}

INSTANTIATE_TEST_SUITE_P(All, LogUniformIntChiSquaredTest,
                         ::testing::ValuesIn(GenParams()), ParamName);

// NOTE: absl::log_uniform_int_distribution is not guaranteed to be stable.
TEST(LogUniformIntDistributionTest, StabilityTest) {
  using testing::ElementsAre;
  // absl::uniform_int_distribution stability relies on
  // absl::random_internal::LeadingSetBit, std::log, std::pow.
  absl::random_internal::sequence_urbg urbg(
      {0x0003eb76f6f7f755ull, 0xFFCEA50FDB2F953Bull, 0xC332DDEFBE6C5AA5ull,
       0x6558218568AB9702ull, 0x2AEF7DAD5B6E2F84ull, 0x1521B62829076170ull,
       0xECDD4775619F1510ull, 0x13CCA830EB61BD96ull, 0x0334FE1EAA0363CFull,
       0xB5735C904C70A239ull, 0xD59E9E0BCBAADE14ull, 0xEECC86BC60622CA7ull});

  std::vector<int> output(6);

  {
    absl::log_uniform_int_distribution<int32_t> dist(0, 256);
    std::generate(std::begin(output), std::end(output),
                  [&] { return dist(urbg); });
    EXPECT_THAT(output, ElementsAre(256, 66, 4, 6, 57, 103));
  }
  urbg.reset();
  {
    absl::log_uniform_int_distribution<int32_t> dist(0, 256, 10);
    std::generate(std::begin(output), std::end(output),
                  [&] { return dist(urbg); });
    EXPECT_THAT(output, ElementsAre(8, 4, 0, 0, 0, 69));
  }
}

}  // namespace
