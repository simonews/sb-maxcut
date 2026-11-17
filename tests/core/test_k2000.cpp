// K2000 (data/raw/k2000/WK2000_1.rud): the complete graph with 2000 nodes and weights +-1, read
// with sbmc::read_rudy. The file is above the 5 MB limit of the repository and is not in git: when
// it is missing, the program prints "SKIP: run scripts/fetch_data.sh --k2000" and exits with 77
// (SKIP_RETURN_CODE) without running the cases.
//
// Reference values from cldArchitect: n, m, W and the counts of the weights; the bits of auto_c0
// were obtained with the library of F04-R1 in the cpu-strict preset. "Strict" means
// SBMC_STRICT_FP_BUILD = 1: there auto_c0 is compared bit for bit, elsewhere with a relative error
// <= 1e-6.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

#include <sbmc/instance.hpp>
#include <sbmc/rudy.hpp>
#include <sbmc/sb.hpp>

#ifndef SBMC_DATA_DIR
#error "SBMC_DATA_DIR must be defined by CMake (absolute path of data/)"
#endif
#ifndef SBMC_STRICT_FP_BUILD
#error "SBMC_STRICT_FP_BUILD must be defined by CMake (1 in the strict presets, 0 otherwise)"
#endif

namespace {

constexpr bool kStrict = SBMC_STRICT_FP_BUILD != 0;
constexpr int kExitSkip = 77;

std::filesystem::path k2000_file() {
  return std::filesystem::path(SBMC_DATA_DIR) / "raw" / "k2000" / "WK2000_1.rud";
}

// The graph is read once (1,999,000 edges) and shared by the cases.
const sbmc::Graph& k2000() {
  static const sbmc::Graph graph = sbmc::read_rudy(k2000_file());
  return graph;
}

std::string hex32(std::uint32_t value) {
  char text[16];
  std::snprintf(text, sizeof(text), "0x%08X", static_cast<unsigned>(value));
  return text;
}

}  // namespace

TEST_CASE("K2000: n, m, W and the weights of WK2000_1.rud") {
  const sbmc::Graph& g = k2000();
  CHECK(g.n == 2000);
  CHECK(g.edges.size() == 1999000u);
  CHECK(sbmc::total_weight(g) == -1040);
  std::int64_t plus = 0;
  std::int64_t minus = 0;
  std::int64_t other = 0;
  for (const sbmc::Edge& e : g.edges) {
    plus += e.w == 1 ? 1 : 0;
    minus += e.w == -1 ? 1 : 0;
    other += e.w != 1 && e.w != -1 ? 1 : 0;
  }
  CHECK(plus == 998980);
  CHECK(minus == 1000020);
  CHECK(other == 0);
}

TEST_CASE("K2000: auto_c0") {
  constexpr std::uint32_t kExpected = 0x3C372DC0u;
  const float c0 = sbmc::auto_c0(k2000());
  INFO("expected " << hex32(kExpected) << ", got " << hex32(std::bit_cast<std::uint32_t>(c0))
                   << " (strict = " << kStrict << ")");
  if (kStrict) {
    CHECK(std::bit_cast<std::uint32_t>(c0) == kExpected);
  } else {
    const double reference = std::bit_cast<float>(kExpected);
    CHECK(std::abs(static_cast<double>(c0) - reference) <= 1e-6 * reference);
  }
}

int main(int argc, char** argv) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(k2000_file(), error)) {
    std::cout << "SKIP: run scripts/fetch_data.sh --k2000\n";
    return kExitSkip;
  }
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  return context.run();
}
