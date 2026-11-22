// sbmc/instance.hpp: T1 (Ising-cut mapping on the 4-node example), int64 sums and validation, T5
// (dense INT8 rows of a Graph), T12 (implicit dense instances) and, since phase 07, the unchecked
// cut of the final check of the solvers (sbmc::detail), against cut_direct as the oracle, with
// the product form of the dense rows since F07-R2.
//
// The T12 reference values were computed by cldArchitect with an independent implementation,
// checked on the Random123 vectors. Random inputs come from std::mt19937_64 with fixed seeds and
// serve only as inputs: their checks are identities, not reference values.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/rudy.hpp>
#include <sbmc/sa.hpp>

#ifndef SBMC_TEST_DATA_DIR
#error "SBMC_TEST_DATA_DIR must be defined by CMake (absolute path of tests/data)"
#endif
#ifndef SBMC_DATA_DIR
#error "SBMC_DATA_DIR must be defined by CMake (absolute path of data/)"
#endif

namespace {

using sbmc::DenseSpinGlass;
using sbmc::Edge;
using sbmc::Graph;
using sbmc::Instance;
using sbmc::InvalidArgument;
using sbmc::Spin;

constexpr std::int64_t kTwoTo31 = std::int64_t{1} << 31;

// The 4-node example of the reference document, A, B, C, D = 0, 1, 2, 3, with the edges AB, BC,
// CD, DA, AC of weight 1 in canonical order.
Graph four_nodes() {
  return Graph{4, {{0, 1, 1}, {0, 2, 1}, {0, 3, 1}, {1, 2, 1}, {2, 3, 1}}};
}

// Spins "with step 3": s_i = +1 if i mod 3 == 0, -1 otherwise.
std::vector<Spin> step3_spins(std::int64_t n) {
  std::vector<Spin> s(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    s[static_cast<std::size_t>(i)] = i % 3 == 0 ? Spin{1} : Spin{-1};
  }
  return s;
}

std::vector<Spin> random_spins(std::int64_t n, std::mt19937_64& rng) {
  std::vector<Spin> s(static_cast<std::size_t>(n));
  for (Spin& spin : s) {
    spin = (rng() & 1u) != 0 ? Spin{1} : Spin{-1};
  }
  return s;
}

// Non-zero weight in [-max_abs, max_abs].
std::int32_t random_weight(std::int32_t max_abs, std::mt19937_64& rng) {
  std::uniform_int_distribution<std::int32_t> magnitude(1, max_abs);
  const std::int32_t w = magnitude(rng);
  return (rng() & 1u) != 0 ? w : -w;
}

// Canonical graph with n nodes and edge_count distinct edges chosen at random.
Graph random_graph(std::int32_t n, std::size_t edge_count, std::int32_t max_abs,
                   std::mt19937_64& rng) {
  std::vector<std::pair<std::int32_t, std::int32_t>> pairs;
  for (std::int32_t u = 0; u < n; ++u) {
    for (std::int32_t v = u + 1; v < n; ++v) {
      pairs.emplace_back(u, v);
    }
  }
  std::shuffle(pairs.begin(), pairs.end(), rng);
  pairs.resize(std::min(edge_count, pairs.size()));
  std::sort(pairs.begin(), pairs.end());
  Graph g;
  g.n = n;
  for (const auto& [u, v] : pairs) {
    g.edges.push_back(Edge{u, v, random_weight(max_abs, rng)});
  }
  return g;
}

bool same_edges(const std::vector<Edge>& a, const std::vector<Edge>& b) {
  return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](const Edge& x, const Edge& y) {
    return x.u == y.u && x.v == y.v && x.w == y.w;
  });
}

std::size_t index(std::int64_t row, std::int64_t col, std::int64_t ld) {
  return static_cast<std::size_t>(row * ld + col);
}

// All rows [0, ld) of the J of a dense instance, with stride ld.
std::vector<std::int8_t> all_rows(const DenseSpinGlass& g, std::int64_t ld) {
  std::vector<std::int8_t> rows(static_cast<std::size_t>(ld * ld));
  sbmc::dense_rows_int8(g, 0, ld, ld, rows);
  return rows;
}

// Restores the OpenMP settings that the thread-count tests change, also when a check throws.
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

// --- T1 ----------------------------------------------------------------------------------------

TEST_CASE("T1: 4-node example: W = 5, E = -3 <-> C = 4, E = -1 <-> C = 3, E = 5 <-> C = 0") {
  const Graph g = four_nodes();
  const Instance inst = g;
  CHECK(sbmc::num_spins(g) == 4);
  CHECK(sbmc::num_spins(inst) == 4);
  CHECK(sbmc::total_weight(g) == 5);
  CHECK(sbmc::total_weight(inst) == 5);

  struct Case {
    std::vector<Spin> s;
    std::int64_t energy;
    std::int64_t cut;
  };
  const std::vector<Case> cases = {
      {{1, -1, 1, -1}, -3, 4},
      {{1, 1, -1, -1}, -1, 3},
      {{1, 1, 1, 1}, 5, 0},
  };
  for (const Case& c : cases) {
    CHECK(sbmc::ising_energy(g, c.s) == c.energy);
    CHECK(sbmc::cut_direct(g, c.s) == c.cut);
    CHECK(sbmc::ising_energy(inst, c.s) == c.energy);
    CHECK(sbmc::cut_direct(inst, c.s) == c.cut);
  }
}

// --- Sums and validation -----------------------------------------------------------------------

TEST_CASE("Sums: cut_direct = (W - E)/2 on random graphs, with W - E even") {
  std::mt19937_64 rng(0x5EED0001u);
  int graphs = 0;
  for (std::int32_t n = 1; n <= 300; n += 6) {  // 50 graphs, n = 1, 7, ..., 295
    const std::size_t max_edges = static_cast<std::size_t>(n) * static_cast<std::size_t>(n - 1) / 2;
    const std::size_t edge_count = max_edges == 0 ? 0 : rng() % (max_edges + 1);
    const Graph g = random_graph(n, std::min<std::size_t>(edge_count, 4000), 1000, rng);
    const std::int64_t w = sbmc::total_weight(g);
    for (int k = 0; k < 5; ++k) {
      const std::vector<Spin> s = random_spins(n, rng);
      const std::int64_t e = sbmc::ising_energy(g, s);
      const std::int64_t c = sbmc::cut_direct(g, s);
      INFO("n = " << n << ", M = " << g.edges.size() << ", W = " << w << ", E = " << e
                  << ", C = " << c);
      CHECK((w - e) % 2 == 0);
      CHECK(2 * c == w - e);
    }
    ++graphs;
  }
  CHECK(graphs == 50);
}

TEST_CASE("Sums: int64 on K4 with all weights equal to 2^31 - 1") {
  constexpr std::int32_t kW = 2147483647;
  const Graph g{4, {{0, 1, kW}, {0, 2, kW}, {0, 3, kW}, {1, 2, kW}, {1, 3, kW}, {2, 3, kW}}};
  const std::vector<Spin> all_up = {1, 1, 1, 1};
  const std::vector<Spin> alternating = {1, -1, 1, -1};
  CHECK(sbmc::total_weight(g) == 12884901882);
  CHECK(sbmc::ising_energy(g, all_up) == 12884901882);
  CHECK(sbmc::cut_direct(g, all_up) == 0);
  CHECK(sbmc::ising_energy(g, alternating) == -4294967294);
  CHECK(sbmc::cut_direct(g, alternating) == 8589934588);
}

TEST_CASE("Sums: an invalid Graph throws InvalidArgument from every function") {
  struct Case {
    std::string name;
    Graph g;
  };
  const std::vector<Case> cases = {
      {"n = 0", Graph{0, {}}},
      {"n = 2^31", Graph{kTwoTo31, {}}},
      {"edge (1, 1)", Graph{4, {{1, 1, 1}}}},
      {"edge (2, 1)", Graph{4, {{2, 1, 1}}}},
      {"v = n", Graph{4, {{0, 4, 1}}}},
      {"w = 0", Graph{4, {{0, 1, 0}}}},
      {"w = INT32_MIN", Graph{4, {{0, 1, std::numeric_limits<std::int32_t>::min()}}}},
      {"(0, 2) before (0, 1)", Graph{4, {{0, 2, 1}, {0, 1, 1}}}},
      {"repeated edge", Graph{4, {{0, 1, 1}, {0, 1, 1}}}},
  };
  const std::vector<Spin> s = {1, -1, 1, -1};
  for (const Case& c : cases) {
    INFO(c.name);
    const Instance inst = c.g;
    CHECK_THROWS_AS(sbmc::total_weight(c.g), InvalidArgument);
    CHECK_THROWS_AS(sbmc::ising_energy(c.g, s), InvalidArgument);
    CHECK_THROWS_AS(sbmc::cut_direct(c.g, s), InvalidArgument);
    CHECK_THROWS_AS(sbmc::total_weight(inst), InvalidArgument);
    CHECK_THROWS_AS(sbmc::ising_energy(inst, s), InvalidArgument);
    CHECK_THROWS_AS(sbmc::cut_direct(inst, s), InvalidArgument);
    std::vector<std::int8_t> out(16, 85);
    CHECK_THROWS_AS(sbmc::dense_rows_int8(c.g, 0, 4, 4, out), InvalidArgument);
    CHECK_THROWS_AS(sbmc::dense_rows_int8(inst, 0, 4, 4, out), InvalidArgument);
    CHECK(std::all_of(out.begin(), out.end(), [](std::int8_t x) { return x == 85; }));
  }
  CHECK_THROWS_AS(sbmc::num_spins(Graph{0, {}}), InvalidArgument);
  CHECK_THROWS_AS(sbmc::num_spins(Graph{kTwoTo31, {}}), InvalidArgument);
  CHECK(sbmc::num_spins(Graph{kTwoTo31 - 1, {}}) == kTwoTo31 - 1);
}

TEST_CASE("Sums: an invalid spin vector throws InvalidArgument") {
  const Graph g = four_nodes();
  const Instance inst = g;
  const DenseSpinGlass dense{4, 1};
  struct Case {
    std::string name;
    std::vector<Spin> s;
  };
  const std::vector<Case> cases = {
      {"length n - 1", {1, -1, 1}},
      {"length n + 1", {1, -1, 1, -1, 1}},
      {"a spin 0", {1, 0, 1, -1}},
      {"a spin 2", {1, -1, 2, -1}},
  };
  for (const Case& c : cases) {
    INFO(c.name);
    CHECK_THROWS_AS(sbmc::ising_energy(g, c.s), InvalidArgument);
    CHECK_THROWS_AS(sbmc::cut_direct(g, c.s), InvalidArgument);
    CHECK_THROWS_AS(sbmc::ising_energy(inst, c.s), InvalidArgument);
    CHECK_THROWS_AS(sbmc::cut_direct(inst, c.s), InvalidArgument);
    CHECK_THROWS_AS(sbmc::ising_energy(dense, c.s), InvalidArgument);
    CHECK_THROWS_AS(sbmc::cut_direct(dense, c.s), InvalidArgument);
  }
}

// --- T5 ----------------------------------------------------------------------------------------

TEST_CASE("T5: dense INT8 rows of a Graph, with padding") {
  constexpr std::int32_t kN = 37;
  constexpr std::int64_t kLd = 48;
  constexpr std::size_t kRowsBytes = static_cast<std::size_t>(kLd * kLd);
  constexpr std::int8_t kSentinel = 85;
  std::mt19937_64 rng(0x5EED0005u);
  const Graph g = random_graph(kN, 250, 127, rng);
  REQUIRE(g.edges.size() == 250u);

  std::vector<std::int8_t> buffer(kRowsBytes + 16, kSentinel);
  sbmc::dense_rows_int8(g, 0, kLd, kLd, buffer);

  std::vector<std::int8_t> expected(kRowsBytes, 0);
  for (const Edge& e : g.edges) {
    expected[index(e.u, e.v, kLd)] = static_cast<std::int8_t>(-e.w);
    expected[index(e.v, e.u, kLd)] = static_cast<std::int8_t>(-e.w);
  }

  SUBCASE("J = -w on the edges, symmetric, zero diagonal, zero padding, tail untouched") {
    int wrong_edges = 0;
    for (const Edge& e : g.edges) {
      wrong_edges += buffer[index(e.u, e.v, kLd)] != -e.w ? 1 : 0;
    }
    CHECK(wrong_edges == 0);
    int asymmetric = 0;
    int nonzero_diagonal = 0;
    int nonzero_padding = 0;
    for (std::int64_t i = 0; i < kLd; ++i) {
      nonzero_diagonal += buffer[index(i, i, kLd)] != 0 ? 1 : 0;
      for (std::int64_t j = 0; j < kLd; ++j) {
        asymmetric += buffer[index(i, j, kLd)] != buffer[index(j, i, kLd)] ? 1 : 0;
        if (i >= kN || j >= kN) {
          nonzero_padding += buffer[index(i, j, kLd)] != 0 ? 1 : 0;
        }
      }
    }
    CHECK(asymmetric == 0);
    CHECK(nonzero_diagonal == 0);
    CHECK(nonzero_padding == 0);
    CHECK(std::equal(expected.begin(), expected.end(), buffer.begin()));
    CHECK(std::all_of(buffer.begin() + static_cast<std::ptrdiff_t>(kRowsBytes), buffer.end(),
                      [](std::int8_t x) { return x == kSentinel; }));
  }

  SUBCASE("rows written in blocks [0, 5), [5, 40), [40, 48) equal the rows written at once") {
    std::vector<std::int8_t> blocks(kRowsBytes, kSentinel);
    const std::span<std::int8_t> all(blocks);
    const std::pair<std::int64_t, std::int64_t> ranges[] = {{0, 5}, {5, 40}, {40, 48}};
    for (const auto& [first_row, last_row] : ranges) {
      sbmc::dense_rows_int8(
          g, first_row, last_row, kLd,
          all.subspan(index(first_row, 0, kLd), index(last_row - first_row, 0, kLd)));
    }
    CHECK(std::equal(blocks.begin(), blocks.end(), buffer.begin()));
  }

  SUBCASE("round trip: the non-zero elements i < j give back the edges, with w = -J") {
    std::vector<Edge> edges;
    for (std::int32_t i = 0; i < kN; ++i) {
      for (std::int32_t j = i + 1; j < kN; ++j) {
        const std::int8_t value = buffer[index(i, j, kLd)];
        if (value != 0) {
          edges.push_back(Edge{i, j, -value});
        }
      }
    }
    CHECK(same_edges(edges, g.edges));
  }

  SUBCASE("InvalidArgument leaves the output unchanged") {
    std::vector<std::int8_t> out(kRowsBytes + 16);
    for (std::size_t k = 0; k < out.size(); ++k) {
      out[k] = static_cast<std::int8_t>(k % 251);
    }
    const std::vector<std::int8_t> original = out;
    const Graph w128{4, {{0, 1, 1}, {1, 2, 128}}};
    const Graph w_minus128{4, {{0, 1, -128}}};
    CHECK_THROWS_AS(sbmc::dense_rows_int8(w128, 0, 4, 4, out), InvalidArgument);
    CHECK_THROWS_AS(sbmc::dense_rows_int8(w_minus128, 0, 4, 4, out), InvalidArgument);
    CHECK_THROWS_AS(sbmc::dense_rows_int8(g, 0, kN - 1, kN - 1, out),
                    InvalidArgument);                                                 // ld = n - 1
    CHECK_THROWS_AS(sbmc::dense_rows_int8(g, 0, 1, kTwoTo31, out), InvalidArgument);  // ld = 2^31
    CHECK_THROWS_AS(sbmc::dense_rows_int8(g, 5, 4, kLd, out), InvalidArgument);       // begin > end
    CHECK_THROWS_AS(sbmc::dense_rows_int8(g, 0, kLd + 1, kLd, out), InvalidArgument);  // end > ld
    CHECK_THROWS_AS(sbmc::dense_rows_int8(g, -1, 4, kLd, out), InvalidArgument);       // begin < 0
    CHECK_THROWS_AS(sbmc::dense_rows_int8(g, 0, kLd, kLd, std::span(out).first(kRowsBytes - 1)),
                    InvalidArgument);  // out shorter than (row_end - row_begin) * ld
    CHECK(out == original);
    // The same shape with |w| = 127 is accepted.
    const Graph w127{4, {{0, 1, 127}, {1, 2, -127}}};
    CHECK_NOTHROW(sbmc::dense_rows_int8(w127, 0, 4, 4, out));
    CHECK(static_cast<int>(out[1]) == -127);
    CHECK(static_cast<int>(out[6]) == 127);
  }
}

// --- T12 ---------------------------------------------------------------------------------------

TEST_CASE("T12: coupling reproduces the reference values and is symmetric") {
  struct Case {
    std::uint64_t seed;
    std::int64_t n;
    std::int64_t i;
    std::int64_t j;
    int expected;
  };
  const std::vector<Case> cases = {
      {1, 64, 0, 1, 1},
      {1, 64, 1, 0, 1},
      {1, 64, 0, 2, 1},
      {1, 64, 5, 3, 1},
      {1, 64, 3, 5, 1},
      {1, 64, 63, 62, -1},
      {0x9E3779B97F4A7C15u, 2001, 1000, 2000, 1},
      {3, 2147483647, 2147483645, 2147483646, 1},
      {3, 2147483647, 0, 2147483646, -1},
      {3, 2147483647, 2147483646, 1, -1},
      {3, 160000, 159998, 159999, 1},
      {3, 160000, 159999, 0, -1},
  };
  for (const Case& c : cases) {
    const DenseSpinGlass g{c.n, c.seed};
    INFO("seed " << c.seed << ", n " << c.n << ", (i, j) = (" << c.i << ", " << c.j << ")");
    CHECK(static_cast<int>(sbmc::coupling(g, c.i, c.j)) == c.expected);
    CHECK(sbmc::coupling(g, c.j, c.i) == sbmc::coupling(g, c.i, c.j));
    CHECK(static_cast<int>(sbmc::coupling(g, c.i, c.i)) == 0);
  }
}

TEST_CASE("T12: invalid dense instances and indices throw InvalidArgument") {
  const DenseSpinGlass g{64, 1};
  CHECK_THROWS_AS(sbmc::coupling(g, -1, 0), InvalidArgument);
  CHECK_THROWS_AS(sbmc::coupling(g, 0, -1), InvalidArgument);
  CHECK_THROWS_AS(sbmc::coupling(g, 64, 0), InvalidArgument);
  CHECK_THROWS_AS(sbmc::coupling(g, 0, 64), InvalidArgument);
  CHECK_NOTHROW(sbmc::coupling(g, 63, 0));

  const std::vector<Spin> s = {1, -1, 1, -1};
  for (const std::int64_t n : {std::int64_t{0}, kTwoTo31}) {
    INFO("n = " << n);
    const DenseSpinGlass bad{n, 1};
    const Instance inst = bad;
    std::vector<std::int8_t> out(16, 85);
    CHECK_THROWS_AS(sbmc::coupling(bad, 0, 0), InvalidArgument);
    CHECK_THROWS_AS(sbmc::dense_rows_int8(bad, 0, 4, 4, out), InvalidArgument);
    CHECK_THROWS_AS(sbmc::dense_rows_int8(inst, 0, 4, 4, out), InvalidArgument);
    CHECK_THROWS_AS(sbmc::total_weight(bad), InvalidArgument);
    CHECK_THROWS_AS(sbmc::total_weight(inst), InvalidArgument);
    CHECK_THROWS_AS(sbmc::num_spins(bad), InvalidArgument);
    CHECK_THROWS_AS(sbmc::ising_energy(bad, s), InvalidArgument);
    CHECK_THROWS_AS(sbmc::cut_direct(bad, s), InvalidArgument);
    CHECK(std::all_of(out.begin(), out.end(), [](std::int8_t x) { return x == 85; }));
  }

  // Shape errors leave the output unchanged.
  std::vector<std::int8_t> out(64 * 64, 85);
  CHECK_THROWS_AS(sbmc::dense_rows_int8(g, 0, 63, 63, out), InvalidArgument);  // ld = n - 1
  CHECK_THROWS_AS(sbmc::dense_rows_int8(g, 2, 1, 64, out), InvalidArgument);
  CHECK_THROWS_AS(sbmc::dense_rows_int8(g, 0, 65, 64, out), InvalidArgument);
  CHECK_THROWS_AS(sbmc::dense_rows_int8(g, 0, 64, 64, std::span(out).first(64 * 64 - 1)),
                  InvalidArgument);
  CHECK(std::all_of(out.begin(), out.end(), [](std::int8_t x) { return x == 85; }));
}

TEST_CASE("T12: n = 64, seed 1: sums and rows") {
  const DenseSpinGlass g{64, 1};
  const std::vector<Spin> s = step3_spins(64);
  CHECK(sbmc::num_spins(g) == 64);
  CHECK(sbmc::total_weight(g) == -16);
  CHECK(sbmc::ising_energy(g, s) == 92);
  CHECK(sbmc::cut_direct(g, s) == -54);

  const std::vector<std::int8_t> rows = all_rows(g, 64);
  const std::vector<int> row0 = {0, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1};
  const std::vector<int> row63 = {-1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 0};
  for (std::int64_t j = 0; j < 16; ++j) {
    INFO("j = " << j);
    CHECK(static_cast<int>(rows[index(0, j, 64)]) == row0[static_cast<std::size_t>(j)]);
    CHECK(static_cast<int>(rows[index(63, 48 + j, 64)]) == row63[static_cast<std::size_t>(j)]);
  }
}

TEST_CASE("T12: n = 1031, seed 7, ld = 1056: rows, padding and sums") {
  constexpr std::int64_t kN = 1031;
  constexpr std::int64_t kLd = 1056;
  const DenseSpinGlass g{kN, 7};
  const std::vector<std::int8_t> rows = all_rows(g, kLd);

  std::int64_t not_coupling = 0;
  std::int64_t asymmetric = 0;
  std::int64_t nonzero_diagonal = 0;
  std::int64_t not_unit = 0;
  std::int64_t nonzero_padding = 0;
  std::int64_t upper = 0;
  std::int64_t upper_plus = 0;
  for (std::int64_t i = 0; i < kLd; ++i) {
    for (std::int64_t j = 0; j < kLd; ++j) {
      const int value = rows[index(i, j, kLd)];
      asymmetric += value != rows[index(j, i, kLd)] ? 1 : 0;
      if (i >= kN || j >= kN) {
        nonzero_padding += value != 0 ? 1 : 0;
        continue;
      }
      not_coupling += value != sbmc::coupling(g, i, j) ? 1 : 0;
      if (i == j) {
        nonzero_diagonal += value != 0 ? 1 : 0;
      } else {
        not_unit += value != 1 && value != -1 ? 1 : 0;
      }
      if (i < j) {
        ++upper;
        upper_plus += value == 1 ? 1 : 0;
      }
    }
  }
  CHECK(not_coupling == 0);
  CHECK(asymmetric == 0);
  CHECK(nonzero_diagonal == 0);
  CHECK(not_unit == 0);
  CHECK(nonzero_padding == 0);
  CHECK(upper == 530965);
  CHECK(upper_plus == 265895);

  const std::vector<Spin> s = step3_spins(kN);
  CHECK(sbmc::total_weight(g) == -825);
  CHECK(sbmc::ising_energy(g, s) == 127);
  CHECK(sbmc::cut_direct(g, s) == -476);
}

TEST_CASE("T12: n = 2000, seeds 11 and 12: sums and statistics") {
  constexpr std::int64_t kN = 2000;
  const DenseSpinGlass g11{kN, 11};
  const DenseSpinGlass g12{kN, 12};
  const std::vector<Spin> s = step3_spins(kN);
  CHECK(sbmc::total_weight(g11) == 584);
  CHECK(sbmc::ising_energy(g11, s) == 306);
  CHECK(sbmc::cut_direct(g11, s) == 139);

  const std::vector<std::int8_t> j11 = all_rows(g11, kN);
  const std::vector<std::int8_t> j12 = all_rows(g12, kN);
  std::int64_t pairs = 0;
  std::int64_t plus = 0;
  std::int64_t different = 0;
  for (std::int64_t i = 0; i < kN; ++i) {
    for (std::int64_t j = i + 1; j < kN; ++j) {
      ++pairs;
      plus += j11[index(i, j, kN)] == 1 ? 1 : 0;
      different += j11[index(i, j, kN)] != j12[index(i, j, kN)] ? 1 : 0;
    }
  }
  CHECK(pairs == 1999000);
  CHECK(plus == 999208);
  const double sigma = 0.5 / std::sqrt(1999000.0);
  CHECK(std::abs(static_cast<double>(plus) / 1999000.0 - 0.5) < 5.0 * sigma);
  CHECK(different == 999404);
  CHECK(std::abs(static_cast<double>(different) - 999500.0) < 5.0 * 0.5 * std::sqrt(1999000.0));

  const std::vector<std::pair<std::int64_t, std::int64_t>> row_counts = {
      {0, 963}, {1, 994}, {999, 1016}, {1998, 944}, {1999, 965}};
  const double sigma_row = 0.5 * std::sqrt(1999.0);
  for (const auto& [row, expected] : row_counts) {
    std::int64_t row_plus = 0;
    for (std::int64_t j = 0; j < kN; ++j) {
      row_plus += j != row && j11[index(row, j, kN)] == 1 ? 1 : 0;
    }
    INFO("row " << row);
    CHECK(row_plus == expected);
    CHECK(std::abs(static_cast<double>(row_plus) - 999.5) < 5.0 * sigma_row);
  }

  const std::vector<std::pair<std::int64_t, std::int64_t>> row_pairs = {
      {0, 1}, {1000, 1001}, {1998, 1999}};
  const std::vector<std::int64_t> expected_dots = {24, -20, 68};
  for (std::size_t p = 0; p < row_pairs.size(); ++p) {
    const auto [a, b] = row_pairs[p];
    std::int64_t dot = 0;
    for (std::int64_t j = 0; j < kN; ++j) {
      if (j != a && j != b) {
        dot += std::int64_t{j11[index(a, j, kN)]} * j11[index(b, j, kN)];
      }
    }
    INFO("rows " << a << " and " << b);
    CHECK(dot == expected_dots[p]);
    CHECK(std::abs(static_cast<double>(dot)) <= 5.0 * std::sqrt(1998.0));
  }
}

TEST_CASE("T12: n = 160000, seed 3: rows 159998 and 159999") {
  constexpr std::int64_t kN = 160000;
  const DenseSpinGlass g{kN, 3};
  std::vector<std::int8_t> rows(static_cast<std::size_t>(2 * kN));
  sbmc::dense_rows_int8(g, kN - 2, kN, kN, rows);

  struct RowCheck {
    std::int64_t row;
    std::int64_t plus;
    std::int64_t minus;
    std::vector<int> first8;
    std::vector<int> last8;
  };
  const std::vector<RowCheck> checks = {
      {159998, 80027, 79972, {1, 1, 1, -1, -1, -1, -1, -1}, {-1, 1, 1, -1, -1, -1, 0, 1}},
      {159999, 79836, 80163, {-1, -1, -1, 1, 1, 1, -1, 1}, {-1, -1, 1, -1, 1, 1, 1, 0}},
  };
  for (const RowCheck& c : checks) {
    INFO("row " << c.row);
    const std::int64_t local = c.row - (kN - 2);
    std::int64_t plus = 0;
    std::int64_t minus = 0;
    std::int64_t not_coupling = 0;
    for (std::int64_t j = 0; j < kN; ++j) {
      const int value = rows[index(local, j, kN)];
      plus += value == 1 ? 1 : 0;
      minus += value == -1 ? 1 : 0;
      not_coupling += value != sbmc::coupling(g, c.row, j) ? 1 : 0;
    }
    CHECK(not_coupling == 0);
    CHECK(plus == c.plus);
    CHECK(minus == c.minus);
    for (std::int64_t t = 0; t < 8; ++t) {
      CHECK(static_cast<int>(rows[index(local, t, kN)]) == c.first8[static_cast<std::size_t>(t)]);
      CHECK(static_cast<int>(rows[index(local, kN - 8 + t, kN)]) ==
            c.last8[static_cast<std::size_t>(t)]);
    }
  }
}

TEST_CASE("T12: the results do not depend on the number of OpenMP threads") {
  constexpr std::int64_t kN = 1500;
  const DenseSpinGlass g{kN, 0xC0FFEEu};
  std::mt19937_64 rng(0x5EED0012u);
  const std::vector<Spin> s = random_spins(kN, rng);

  const OmpSettingsGuard guard;
  omp_set_dynamic(0);
  const int initial_threads = guard.initial_max_threads();

  struct Results {
    std::vector<std::int8_t> rows;
    std::int64_t w = 0;
    std::int64_t e = 0;
    std::int64_t c = 0;
  };
  std::vector<Results> results;
  for (const int threads : {1, 3, initial_threads}) {
    omp_set_num_threads(threads);
    REQUIRE(omp_get_max_threads() == threads);
    Results r;
    r.rows = all_rows(g, kN);
    r.w = sbmc::total_weight(g);
    r.e = sbmc::ising_energy(g, s);
    r.c = sbmc::cut_direct(g, s);
    results.push_back(std::move(r));
  }
  for (std::size_t k = 1; k < results.size(); ++k) {
    INFO("configuration " << k << " against the single-thread one");
    CHECK(results[k].rows == results[0].rows);
    CHECK(results[k].w == results[0].w);
    CHECK(results[k].e == results[0].e);
    CHECK(results[k].c == results[0].c);
  }
  CHECK(2 * results[0].c == results[0].w - results[0].e);

  omp_set_dynamic(guard.initial_dynamic());
  omp_set_num_threads(initial_threads);
  CHECK(omp_get_max_threads() == initial_threads);
}

TEST_CASE("T12: n = 300: W, E and C equal double loops over coupling()") {
  constexpr std::int64_t kN = 300;
  const DenseSpinGlass g{kN, 0x243F6A8885A308D3u};
  std::mt19937_64 rng(0x5EED0300u);
  const std::vector<Spin> s = random_spins(kN, rng);
  std::int64_t w = 0;
  std::int64_t e = 0;
  std::int64_t c = 0;
  for (std::int64_t i = 0; i < kN; ++i) {
    for (std::int64_t j = i + 1; j < kN; ++j) {
      const std::int64_t w_ij = -sbmc::coupling(g, i, j);
      const auto si = s[static_cast<std::size_t>(i)];
      const auto sj = s[static_cast<std::size_t>(j)];
      w += w_ij;
      e += w_ij * si * sj;
      c += si != sj ? w_ij : 0;
    }
  }
  CHECK(sbmc::total_weight(g) == w);
  CHECK(sbmc::ising_energy(g, s) == e);
  CHECK(sbmc::cut_direct(g, s) == c);
  CHECK(2 * c == w - e);
}

TEST_CASE("T12: the Instance versions give the same values as the DenseSpinGlass versions") {
  constexpr std::int64_t kN = 300;
  constexpr std::int64_t kLd = 304;
  const DenseSpinGlass g{kN, 5};
  const Instance inst = g;
  std::mt19937_64 rng(0x5EED0301u);
  const std::vector<Spin> s = random_spins(kN, rng);
  CHECK(sbmc::num_spins(inst) == sbmc::num_spins(g));
  CHECK(sbmc::total_weight(inst) == sbmc::total_weight(g));
  CHECK(sbmc::ising_energy(inst, s) == sbmc::ising_energy(g, s));
  CHECK(sbmc::cut_direct(inst, s) == sbmc::cut_direct(g, s));
  std::vector<std::int8_t> from_inst(static_cast<std::size_t>(kLd * kLd), 85);
  std::vector<std::int8_t> from_dense(static_cast<std::size_t>(kLd * kLd), 85);
  sbmc::dense_rows_int8(inst, 0, kLd, kLd, from_inst);
  sbmc::dense_rows_int8(g, 0, kLd, kLd, from_dense);
  CHECK(from_inst == from_dense);
  // Unaligned row ranges give the same rows as one call over all rows.
  std::vector<std::int8_t> pieces(static_cast<std::size_t>(kLd * kLd), 85);
  const std::span<std::int8_t> all(pieces);
  const std::pair<std::int64_t, std::int64_t> ranges[] = {
      {0, 3}, {3, 3}, {3, 150}, {150, 299}, {299, 304}};
  for (const auto& [first_row, last_row] : ranges) {
    sbmc::dense_rows_int8(
        g, first_row, last_row, kLd,
        all.subspan(index(first_row, 0, kLd), index(last_row - first_row, 0, kLd)));
  }
  CHECK(pieces == from_dense);
}

// --- sbmc::detail: the cut of the final check of the solvers (phase 07) ----------------------

namespace {

Graph test_rudy(const char* name) {
  return sbmc::read_rudy(std::string(SBMC_TEST_DATA_DIR) + "/" + name);
}

// Spin vector r of the detail cases: s_i = sa_initial_spin(9, r, i).
std::vector<Spin> stream_spins(std::int64_t n, std::uint32_t r) {
  std::vector<Spin> s(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    s[static_cast<std::size_t>(i)] = sbmc::sa_initial_spin(9, r, static_cast<std::uint32_t>(i));
  }
  return s;
}

// Rows [0, n) of J with stride ld, from dense_rows_int8.
template <typename G>
std::vector<std::int8_t> rows_of(const G& g, std::int64_t ld) {
  std::vector<std::int8_t> rows(static_cast<std::size_t>(g.n * ld));
  sbmc::dense_rows_int8(g, 0, g.n, ld, rows);
  return rows;
}

}  // namespace

TEST_CASE("detail: cut_edges_unchecked equals cut_direct") {
  SUBCASE("four_nodes.rudy, all 16 spin vectors") {
    const Graph g = test_rudy("four_nodes.rudy");
    REQUIRE(g.n == 4);
    for (unsigned mask = 0; mask < 16u; ++mask) {
      std::vector<Spin> s(4);
      for (unsigned i = 0; i < 4u; ++i) {
        s[i] = ((mask >> i) & 1u) != 0 ? Spin{-1} : Spin{1};
      }
      INFO("mask = " << mask);
      CHECK(sbmc::detail::cut_edges_unchecked(g, s) == sbmc::cut_direct(g, s));
    }
    const std::vector<Spin> alternating = {1, -1, 1, -1};
    const std::vector<Spin> halves = {1, 1, -1, -1};
    CHECK(sbmc::detail::cut_edges_unchecked(g, alternating) == 4);
    CHECK(sbmc::detail::cut_edges_unchecked(g, halves) == 3);
  }
  SUBCASE("data/gset/G1, 20 spin vectors") {
    const Graph g = sbmc::read_rudy(std::string(SBMC_DATA_DIR) + "/gset/G1");
    for (std::uint32_t r = 0; r < 20; ++r) {
      const std::vector<Spin> s = stream_spins(g.n, r);
      INFO("r = " << r);
      CHECK(sbmc::detail::cut_edges_unchecked(g, s) == sbmc::cut_direct(g, s));
    }
  }
  SUBCASE("weighted_isolated_13.rudy, 64 spin vectors (sums beyond 2^31)") {
    const Graph g = test_rudy("weighted_isolated_13.rudy");
    bool beyond = false;
    for (std::uint32_t r = 0; r < 64; ++r) {
      const std::vector<Spin> s = stream_spins(g.n, r);
      const std::int64_t expected = sbmc::cut_direct(g, s);
      beyond = beyond || expected >= kTwoTo31 || expected < -kTwoTo31;
      INFO("r = " << r);
      CHECK(sbmc::detail::cut_edges_unchecked(g, s) == expected);
    }
    CHECK(beyond);
  }
  SUBCASE("graphs without edges") {
    const std::vector<Spin> five = {1, -1, 1, -1, -1};
    const std::vector<Spin> one = {-1};
    CHECK(sbmc::detail::cut_edges_unchecked(Graph{5, {}}, five) == 0);
    CHECK(sbmc::detail::cut_edges_unchecked(Graph{1, {}}, one) == 0);
  }
}

TEST_CASE("detail: cut_dense_rows_unchecked equals cut_direct") {
  SUBCASE("DenseSpinGlass{200, 7}, ld = 200 and ld = 256 (padding columns set to 77)") {
    const DenseSpinGlass g{200, 7};
    for (const std::int64_t ld : {std::int64_t{200}, std::int64_t{256}}) {
      std::vector<std::int8_t> rows = rows_of(g, ld);
      for (std::int64_t i = 0; i < g.n; ++i) {
        for (std::int64_t j = g.n; j < ld; ++j) {
          rows[index(i, j, ld)] = 77;  // never read
        }
      }
      for (std::uint32_t r = 0; r < 20; ++r) {
        const std::vector<Spin> s = stream_spins(g.n, r);
        INFO("ld = " << ld << ", r = " << r);
        CHECK(sbmc::detail::cut_dense_rows_unchecked(rows, g.n, ld, s) == sbmc::cut_direct(g, s));
      }
    }
  }
  SUBCASE("DenseSpinGlass{1, 3}: 0") {
    const DenseSpinGlass g{1, 3};
    const std::vector<std::int8_t> rows = rows_of(g, 1);
    const std::vector<Spin> s = {-1};
    CHECK(sbmc::detail::cut_dense_rows_unchecked(rows, 1, 1, s) == 0);
    CHECK(sbmc::cut_direct(g, s) == 0);
  }
  SUBCASE("torus_4x6_pm1.rudy, ld = 24 and ld = 32: the cut of the Graph") {
    const Graph g = test_rudy("torus_4x6_pm1.rudy");
    REQUIRE(g.n == 24);
    for (const std::int64_t ld : {std::int64_t{24}, std::int64_t{32}}) {
      const std::vector<std::int8_t> rows = rows_of(g, ld);
      for (std::uint32_t r = 0; r < 20; ++r) {
        const std::vector<Spin> s = stream_spins(g.n, r);
        INFO("ld = " << ld << ", r = " << r);
        CHECK(sbmc::detail::cut_dense_rows_unchecked(rows, g.n, ld, s) == sbmc::cut_direct(g, s));
      }
    }
  }
}

namespace {

// The spin vectors of the product-form cases: stream_spins with r = 0..7, all +1 and all -1.
std::vector<std::vector<Spin>> product_form_vectors(std::int64_t n) {
  std::vector<std::vector<Spin>> vectors;
  for (std::uint32_t r = 0; r < 8; ++r) {
    vectors.push_back(stream_spins(n, r));
  }
  vectors.emplace_back(static_cast<std::size_t>(n), Spin{1});
  vectors.emplace_back(static_cast<std::size_t>(n), Spin{-1});
  return vectors;
}

// Rows of g with stride ld and the padding columns j >= n set to 77 (never read).
template <typename G>
std::vector<std::int8_t> padded_rows(const G& g, std::int64_t ld) {
  std::vector<std::int8_t> rows = rows_of(g, ld);
  for (std::int64_t i = 0; i < g.n; ++i) {
    for (std::int64_t j = g.n; j < ld; ++j) {
      rows[index(i, j, ld)] = 77;
    }
  }
  return rows;
}

}  // namespace

// Since F07-R2 the cut of the dense rows is (S - R) / 2, with S = sum_i s_i sum_{j>i} J_ij s_j
// and R = sum_{i<j} J_ij (src/core/instance.cpp): the same integer as the masked sum of F07-R1.
TEST_CASE("detail: cut_dense_rows_unchecked, product form") {
  SUBCASE("DenseSpinGlass{2000, 1}, ld = 2000 and ld = 2048 (padding columns set to 77)") {
    const DenseSpinGlass g{2000, 1};
    for (const std::int64_t ld : {std::int64_t{2000}, std::int64_t{2048}}) {
      const std::vector<std::int8_t> rows = padded_rows(g, ld);
      int v = 0;
      for (const std::vector<Spin>& s : product_form_vectors(g.n)) {
        INFO("ld = " << ld << ", vector " << v++);
        CHECK(sbmc::detail::cut_dense_rows_unchecked(rows, g.n, ld, s) == sbmc::cut_direct(g, s));
      }
    }
  }
  SUBCASE("complete Graph n = 300, w = +127 or -127, ld = 320 (padding columns set to 77)") {
    constexpr std::int32_t kN = 300;
    Graph g{kN, {}};
    for (std::int32_t u = 0; u < kN; ++u) {
      for (std::int32_t v = u + 1; v < kN; ++v) {
        g.edges.push_back(Edge{u, v, (u + v) % 2 == 0 ? 127 : -127});
      }
    }
    const std::vector<std::int8_t> rows = padded_rows(g, 320);
    int v = 0;
    for (const std::vector<Spin>& s : product_form_vectors(g.n)) {
      INFO("vector " << v++);
      CHECK(sbmc::detail::cut_dense_rows_unchecked(rows, g.n, 320, s) == sbmc::cut_direct(g, s));
    }
  }
  SUBCASE("DenseSpinGlass{2, 5} and {3, 9}: every spin vector") {
    for (const DenseSpinGlass g : {DenseSpinGlass{2, 5}, DenseSpinGlass{3, 9}}) {
      const std::vector<std::int8_t> rows = rows_of(g, g.n);
      for (unsigned mask = 0; mask < (1u << g.n); ++mask) {
        std::vector<Spin> s(static_cast<std::size_t>(g.n));
        for (std::int64_t i = 0; i < g.n; ++i) {
          s[static_cast<std::size_t>(i)] = ((mask >> i) & 1u) != 0 ? Spin{-1} : Spin{1};
        }
        INFO("n = " << g.n << ", mask = " << mask);
        CHECK(sbmc::detail::cut_dense_rows_unchecked(rows, g.n, g.n, s) == sbmc::cut_direct(g, s));
      }
    }
  }
  SUBCASE("DenseSpinGlass{300, 9}, s_i = +1 for even i and -1 for odd i") {
    const DenseSpinGlass g{300, 9};
    const std::vector<std::int8_t> rows = rows_of(g, g.n);
    std::vector<Spin> s(static_cast<std::size_t>(g.n));
    for (std::int64_t i = 0; i < g.n; ++i) {
      s[static_cast<std::size_t>(i)] = i % 2 == 0 ? Spin{1} : Spin{-1};
    }
    CHECK(sbmc::detail::cut_dense_rows_unchecked(rows, g.n, g.n, s) == sbmc::cut_direct(g, s));
  }
}
