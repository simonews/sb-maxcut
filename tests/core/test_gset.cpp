// T6: reading the G-set files of data/gset/ with sbmc::read_rudy (sbmc/rudy.hpp).
//
// The files are the Stanford originals, byte for byte: header "N M " with a trailing space, edge
// lines with i < j in no particular order. The reference values (n, m, W, first and last
// canonical edges, auto_c0 of G1) come from cldArchitect; the bits of auto_c0(G1) were obtained
// with the library of F04-R1 in the cpu-strict preset. "Strict" means SBMC_STRICT_FP_BUILD = 1:
// there auto_c0 is compared bit for bit, elsewhere with a relative error <= 1e-6.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

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

using sbmc::Graph;

constexpr bool kStrict = SBMC_STRICT_FP_BUILD != 0;

Graph gset(const std::string& name) {
  return sbmc::read_rudy(std::string(SBMC_DATA_DIR) + "/gset/" + name);
}

std::string hex32(std::uint32_t value) {
  char text[16];
  std::snprintf(text, sizeof(text), "0x%08X", static_cast<unsigned>(value));
  return text;
}

struct Expected {
  const char* name;
  std::int64_t n;
  std::int64_t m;
  std::int64_t w;
};

}  // namespace

TEST_CASE("T6: n, m and W of the 13 G-set files of data/gset/") {
  const std::vector<Expected> expected = {
      {"G1", 800, 19176, 19176},   {"G2", 800, 19176, 19176},   {"G11", 800, 1600, 34},
      {"G12", 800, 1600, -4},      {"G14", 800, 4694, 4694},    {"G15", 800, 4661, 4661},
      {"G22", 2000, 19990, 19990}, {"G43", 1000, 9990, 9990},   {"G48", 3000, 6000, 6000},
      {"G55", 5000, 12498, 12498}, {"G63", 7000, 41459, 41459}, {"G70", 10000, 9999, 9999},
      {"G81", 20000, 40000, 34},
  };
  for (const Expected& e : expected) {
    INFO(e.name);
    const Graph g = gset(e.name);
    CHECK(g.n == e.n);
    CHECK(static_cast<std::int64_t>(g.edges.size()) == e.m);
    CHECK(sbmc::total_weight(g) == e.w);
  }
}

TEST_CASE("T6: first and last canonical edges of G1 and G81") {
  const Graph g1 = gset("G1");
  REQUIRE(g1.edges.size() == 19176u);
  CHECK(g1.edges.front().u == 0);
  CHECK(g1.edges.front().v == 15);
  CHECK(g1.edges.front().w == 1);
  CHECK(g1.edges.back().u == 794);
  CHECK(g1.edges.back().v == 797);
  CHECK(g1.edges.back().w == 1);

  const Graph g81 = gset("G81");
  REQUIRE(g81.edges.size() == 40000u);
  CHECK(g81.edges.front().u == 0);
  CHECK(g81.edges.front().v == 1);
  CHECK(g81.edges.front().w == -1);
  CHECK(g81.edges.back().u == 19998);
  CHECK(g81.edges.back().v == 19999);
  CHECK(g81.edges.back().w == -1);
}

TEST_CASE("T6: G1 with every spin +1 has E = 19176 and C = 0") {
  const Graph g1 = gset("G1");
  const std::vector<sbmc::Spin> all_up(static_cast<std::size_t>(g1.n), sbmc::Spin{1});
  CHECK(sbmc::ising_energy(g1, all_up) == 19176);
  CHECK(sbmc::cut_direct(g1, all_up) == 0);
}

TEST_CASE("T6: auto_c0 of G1") {
  constexpr std::uint32_t kExpected = 0x3D98721Fu;
  const float c0 = sbmc::auto_c0(gset("G1"));
  INFO("expected " << hex32(kExpected) << ", got " << hex32(std::bit_cast<std::uint32_t>(c0))
                   << " (strict = " << kStrict << ")");
  if (kStrict) {
    CHECK(std::bit_cast<std::uint32_t>(c0) == kExpected);
  } else {
    const double reference = std::bit_cast<float>(kExpected);
    CHECK(std::abs(static_cast<double>(c0) - reference) <= 1e-6 * reference);
  }
}
