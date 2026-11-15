// Initial positions x0 (sbmc/initial_state.hpp): reference bits, layout, replica split, thread
// independence, range and mean, preconditions. These tests run in every preset: the reference
// bits must come out identical with and without FMA contraction.
//
// The reference values were computed by cldArchitect with an independent implementation,
// checked on the Random123 vectors.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <omp.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/initial_state.hpp>

namespace {

using sbmc::InitialState;
using sbmc::InvalidArgument;

std::string hex32(std::uint32_t value) {
  char text[16];
  std::snprintf(text, sizeof(text), "0x%08X", static_cast<unsigned>(value));
  return text;
}

std::uint32_t bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

// Number of elements whose bits differ (sizes must match).
std::size_t bit_differences(const std::vector<float>& a, const std::vector<float>& b) {
  std::size_t count = a.size() == b.size() ? 0 : 1;
  for (std::size_t k = 0; k < a.size() && k < b.size(); ++k) {
    count += bits(a[k]) != bits(b[k]) ? 1 : 0;
  }
  return count;
}

// Restores the OpenMP settings changed by the thread-count test, also when a check throws.
class OmpSettingsGuard {
 public:
  OmpSettingsGuard() : max_threads_(omp_get_max_threads()), dynamic_(omp_get_dynamic()) {}
  ~OmpSettingsGuard() {
    omp_set_dynamic(dynamic_);
    omp_set_num_threads(max_threads_);
  }
  OmpSettingsGuard(const OmpSettingsGuard&) = delete;
  OmpSettingsGuard& operator=(const OmpSettingsGuard&) = delete;

  [[nodiscard]] int initial_max_threads() const noexcept { return max_threads_; }
  [[nodiscard]] int initial_dynamic() const noexcept { return dynamic_; }

 private:
  int max_threads_;
  int dynamic_;
};

}  // namespace

TEST_CASE("x0: initial_position reproduces the reference bits") {
  struct Case {
    std::uint64_t seed;
    std::uint32_t replica;
    std::uint32_t spin;
    std::uint32_t bits;
  };
  constexpr Case kCases[] = {
      {0x2Au, 0, 0, 0xBD0D967Au},     {0x2Au, 0, 1, 0xBD80106Du},
      {0x2Au, 0, 2, 0xBC23D50Du},     {0x2Au, 0, 3, 0xBDAEBE7Au},
      {0x2Au, 1, 0, 0x3DB94E65u},     {0x2Au, 1023, 149999, 0x3D5F2ACDu},
      {0x0u, 0, 0, 0x3D95193Bu},      {0xFFFFFFFFFFFFFFFFu, 4294967295u, 2147483646u, 0xBDB9EF48u},
      {0x7u, 5, 123456, 0x3D91B880u},
  };
  for (const Case& c : kCases) {
    const std::uint32_t got = bits(sbmc::initial_position(c.seed, c.replica, c.spin));
    INFO("seed " << c.seed << ", replica " << c.replica << ", spin " << c.spin << ": expected "
                 << hex32(c.bits) << ", got " << hex32(got));
    CHECK(got == c.bits);
  }
}

TEST_CASE("x0: make_initial_state(5, 3, 9): fields and replica-major layout") {
  const InitialState state = sbmc::make_initial_state(5, 3, 9);
  CHECK(state.n == 5);
  CHECK(state.replicas == 3);
  CHECK(state.first_replica == 0);
  const std::vector<std::uint32_t> expected = {
      0x3CCB9740u, 0xBBDD4900u, 0xBD474A53u, 0xBCA6E500u, 0x3D8A82A2u,
      0xBD5025EAu, 0x3D38BC7Au, 0x3D8D9F9Fu, 0x3D67E6A0u, 0xBDA0B1EDu,
      0xBDB17475u, 0xBB6A7D33u, 0xBDBCCECAu, 0xBCA6EAA0u, 0xBD8CE1F3u,
  };
  REQUIRE(state.x0.size() == expected.size());
  for (std::size_t k = 0; k < expected.size(); ++k) {
    INFO("x0[" << k << "]: expected " << hex32(expected[k]) << ", got "
               << hex32(bits(state.x0[k])));
    CHECK(bits(state.x0[k]) == expected[k]);
  }
}

TEST_CASE("x0: make_initial_state(1000, 8, 42) has the bits of initial_position") {
  const InitialState state = sbmc::make_initial_state(1000, 8, 42);
  REQUIRE(state.x0.size() == 8000u);
  std::size_t different = 0;
  for (std::uint32_t r = 0; r < 8; ++r) {
    for (std::uint32_t i = 0; i < 1000; ++i) {
      different += bits(state.x0[r * 1000u + i]) != bits(sbmc::initial_position(42, r, i)) ? 1 : 0;
    }
  }
  CHECK(different == 0u);
}

TEST_CASE("x0: splitting the replicas with first_replica gives the same values") {
  const InitialState all = sbmc::make_initial_state(1000, 8, 42);
  const InitialState first = sbmc::make_initial_state(1000, 3, 42, 0);
  const InitialState second = sbmc::make_initial_state(1000, 5, 42, 3);
  CHECK(second.first_replica == 3);
  CHECK(second.replicas == 5);
  std::vector<float> joined = first.x0;
  joined.insert(joined.end(), second.x0.begin(), second.x0.end());
  CHECK(bit_differences(joined, all.x0) == 0u);

  // The last valid global indices: first_replica + replicas = 2^32.
  const InitialState last = sbmc::make_initial_state(6, 3, 77, (std::int64_t{1} << 32) - 3);
  std::size_t different = 0;
  for (std::uint32_t r = 0; r < 3; ++r) {
    for (std::uint32_t i = 0; i < 6; ++i) {
      const auto global = static_cast<std::uint32_t>(0xFFFFFFFDu + r);
      different += bits(last.x0[r * 6u + i]) != bits(sbmc::initial_position(77, global, i)) ? 1 : 0;
    }
  }
  CHECK(different == 0u);
}

TEST_CASE("x0: the result does not depend on the number of OpenMP threads") {
  const OmpSettingsGuard guard;
  omp_set_dynamic(0);
  const int initial_threads = guard.initial_max_threads();
  std::vector<std::vector<float>> results;
  for (const int threads : {1, 3, initial_threads}) {
    omp_set_num_threads(threads);
    REQUIRE(omp_get_max_threads() == threads);
    results.push_back(sbmc::make_initial_state(20000, 16, 7).x0);
  }
  CHECK(results[0].size() == 320000u);
  CHECK(bit_differences(results[1], results[0]) == 0u);
  CHECK(bit_differences(results[2], results[0]) == 0u);
  omp_set_dynamic(guard.initial_dynamic());
  omp_set_num_threads(initial_threads);
  CHECK(omp_get_max_threads() == initial_threads);
}

TEST_CASE("x0: values in [-0.1f, 0.1f) with a mean compatible with 0") {
  constexpr std::uint32_t kCount = 1000000;
  std::size_t outside = 0;
  double sum = 0.0;
  for (std::uint32_t i = 0; i < kCount; ++i) {
    const float x = sbmc::initial_position(42, 0, i);
    outside += x < -0.1f || x >= 0.1f ? 1 : 0;
    sum += static_cast<double>(x);
  }
  CHECK(outside == 0u);
  const double mean = sum / kCount;
  const double sigma_mean = (0.2 / std::sqrt(12.0)) / 1000.0;
  INFO("mean " << mean << ", 5 sigma " << 5.0 * sigma_mean);
  CHECK(std::abs(mean) < 5.0 * sigma_mean);
}

TEST_CASE("x0: invalid arguments throw InvalidArgument") {
  constexpr std::int64_t kTwoTo31 = std::int64_t{1} << 31;
  constexpr std::int64_t kTwoTo32 = std::int64_t{1} << 32;
  CHECK_THROWS_AS(sbmc::make_initial_state(0, 1, 1), InvalidArgument);
  CHECK_THROWS_AS(sbmc::make_initial_state(kTwoTo31, 1, 1), InvalidArgument);
  CHECK_THROWS_AS(sbmc::make_initial_state(4, 0, 1), InvalidArgument);
  CHECK_THROWS_AS(sbmc::make_initial_state(4, -1, 1), InvalidArgument);
  CHECK_THROWS_AS(sbmc::make_initial_state(4, 1, 1, -1), InvalidArgument);
  CHECK_THROWS_AS(sbmc::make_initial_state(4, 3, 1, kTwoTo32 - 2), InvalidArgument);
  CHECK_NOTHROW(sbmc::make_initial_state(4, 2, 1, kTwoTo32 - 2));
}
