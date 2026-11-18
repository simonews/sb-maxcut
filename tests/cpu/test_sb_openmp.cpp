// C1, simulated bifurcation with OpenMP over blocks of replicas (sbmc/sb_openmp.hpp): T7 against
// the constants of C0 (G1-G6, GS-G1) and against C0 itself field by field, the integer
// accumulators at their limits, independence from the number of threads and from the split of
// the replicas, other properties, preconditions.
//
// The constants of G1-G6 and GS-G1 are those of test_sb_sequential.cpp (computed by cldArchitect
// with two independent implementations; optima from an exact enumeration). "Strict" means
// SBMC_STRICT_FP_BUILD = 1 (no FMA contraction): there C1 must give every field of C0 bit for bit
// (decision H3, dSB and bSB); in the production presets only the properties that hold in every
// build are checked. The helper functions are copies of those of test_sb_sequential.cpp.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <omp.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/rudy.hpp>
#include <sbmc/sb.hpp>
#include <sbmc/sb_openmp.hpp>
#include <sbmc/sb_sequential.hpp>

#ifndef SBMC_TEST_DATA_DIR
#error "SBMC_TEST_DATA_DIR must be defined by CMake (absolute path of tests/data)"
#endif
#ifndef SBMC_STRICT_FP_BUILD
#error "SBMC_STRICT_FP_BUILD must be defined by CMake (1 in the strict presets, 0 otherwise)"
#endif
#ifndef SBMC_DATA_DIR
#error "SBMC_DATA_DIR must be defined by CMake (absolute path of data/)"
#endif

namespace {

using sbmc::DenseSpinGlass;
using sbmc::Graph;
using sbmc::Instance;
using sbmc::InvalidArgument;
using sbmc::SbAlgorithm;
using sbmc::SbParams;
using sbmc::SbResult;
using sbmc::Spin;

constexpr bool kStrict = SBMC_STRICT_FP_BUILD != 0;
constexpr int kLanes = sbmc::kSbOpenmpReplicasPerBlock;

std::uint32_t bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

std::string hex32(std::uint32_t value) {
  char text[16];
  std::snprintf(text, sizeof(text), "0x%08X", static_cast<unsigned>(value));
  return text;
}

std::string hex64(std::uint64_t value) {
  char text[24];
  std::snprintf(text, sizeof(text), "0x%016llX", static_cast<unsigned long long>(value));
  return text;
}

Graph rudy(const char* name) {
  return sbmc::read_rudy(std::string(SBMC_TEST_DATA_DIR) + "/" + name);
}

Graph g1_of_gset() {
  return sbmc::read_rudy(std::string(SBMC_DATA_DIR) + "/gset/G1");
}

SbParams params(SbAlgorithm algorithm, int replicas, int steps, float c0, std::uint64_t seed,
                int eval_every) {
  SbParams p;
  p.algorithm = algorithm;
  p.replicas = replicas;
  p.steps = steps;
  p.dt = 1.25f;
  p.a0 = 1.0f;
  p.c0 = c0;
  p.eval_every = eval_every;
  p.seed = seed;
  p.first_replica = 0;
  return p;
}

// FNV-1a 64 of the bytes of final_x (index r * n + i, 4 little-endian bytes per float) followed
// by those of final_y.
std::uint64_t fingerprint(const SbResult& result) {
  std::uint64_t h = 0xCBF29CE484222325ull;
  for (const std::vector<float>* state : {&result.final_x, &result.final_y}) {
    for (const float value : *state) {
      const std::uint32_t word = bits(value);
      for (int byte = 0; byte < 4; ++byte) {
        h ^= (word >> (8 * byte)) & 0xFFu;
        h *= 0x100000001B3ull;
      }
    }
  }
  return h;
}

// FNV-1a 64 of the bytes of trace_x followed by those of trace_y, in the order [k][r][i] (4
// little-endian bytes per float): the fingerprint of the recorded trajectory.
std::uint64_t trajectory_fingerprint(const SbResult& result) {
  std::uint64_t h = 0xCBF29CE484222325ull;
  for (const std::vector<float>* state : {&result.trace_x, &result.trace_y}) {
    for (const float value : *state) {
      const std::uint32_t word = bits(value);
      for (int byte = 0; byte < 4; ++byte) {
        h ^= (word >> (8 * byte)) & 0xFFu;
        h *= 0x100000001B3ull;
      }
    }
  }
  return h;
}

std::span<const Spin> replica_spins(const SbResult& result, int r) {
  const auto n = static_cast<std::size_t>(result.n);
  return {result.best_spins.data() + static_cast<std::size_t>(r) * n, n};
}

std::size_t bit_differences(const std::vector<float>& a, const std::vector<float>& b) {
  std::size_t count = a.size() == b.size() ? 0 : 1;
  for (std::size_t k = 0; k < a.size() && k < b.size(); ++k) {
    count += bits(a[k]) != bits(b[k]) ? 1 : 0;
  }
  return count;
}

bool same_replica_results(const SbResult& a, const SbResult& b) {
  if (a.per_replica.size() != b.per_replica.size()) {
    return false;
  }
  for (std::size_t r = 0; r < a.per_replica.size(); ++r) {
    const auto& x = a.per_replica[r];
    const auto& y = b.per_replica[r];
    if (x.best_energy != y.best_energy || x.best_cut != y.best_cut || x.best_step != y.best_step ||
        x.final_cut != y.final_cut) {
      return false;
    }
  }
  return a.best_spins == b.best_spins;
}

// The fields that differ between two results, threads excluded; empty when they are equal field
// by field (n, replicas, steps, first_replica, the bits of c0, W, per_replica, best_spins, the bits
// of final_x and final_y, trace_steps, the bits of trace_x and trace_y).
std::string field_differences(const SbResult& a, const SbResult& b) {
  std::string differ;
  const auto add = [&differ](bool different, const char* field) {
    if (different) {
      differ += differ.empty() ? field : std::string(", ") + field;
    }
  };
  add(a.n != b.n, "n");
  add(a.replicas != b.replicas, "replicas");
  add(a.steps != b.steps, "steps");
  add(a.first_replica != b.first_replica, "first_replica");
  add(bits(a.c0) != bits(b.c0), "c0");
  add(a.total_weight != b.total_weight, "total_weight");
  add(!same_replica_results(a, b), "per_replica or best_spins");
  add(bit_differences(a.final_x, b.final_x) != 0, "final_x");
  add(bit_differences(a.final_y, b.final_y) != 0, "final_y");
  add(a.trace_steps != b.trace_steps, "trace_steps");
  add(bit_differences(a.trace_x, b.trace_x) != 0, "trace_x");
  add(bit_differences(a.trace_y, b.trace_y) != 0, "trace_y");
  return differ;
}

// Restores the OpenMP settings changed by a test, also when a check throws.
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

// The team of C1: min(T_req, B), with T_req = threads or omp_get_max_threads() for threads = 0.
int expected_team(int threads, int replicas) {
  const int requested = threads > 0 ? threads : omp_get_max_threads();
  return std::min(requested, (replicas + kLanes - 1) / kLanes);
}

// The reference cases G1-G6 (R = 8), with the constants of test_sb_sequential.cpp.
struct ReferenceCase {
  std::string name;
  Instance instance;
  SbAlgorithm algorithm;
  float c0;  // 0: auto
  int steps;
  std::uint64_t seed;
  int eval_every;
  std::int64_t optimum;
  // Strict values.
  std::vector<int> best_step;
  std::vector<std::int64_t> final_cut;
  std::uint64_t fingerprint;
  std::uint32_t c0_bits;
  std::int64_t total_weight;
  std::uint64_t trajectory;
};

std::vector<ReferenceCase> reference_cases() {
  return {
      {.name = "G1",
       .instance = rudy("four_nodes.rudy"),
       .algorithm = SbAlgorithm::discrete,
       .c0 = 0.3f,
       .steps = 100,
       .seed = 2,
       .eval_every = 0,
       .optimum = 4,
       .best_step = {4, 3, 4, 7, 4, 3, 3, 7},
       .final_cut = {4, 4, 4, 4, 4, 4, 4, 4},
       .fingerprint = 0x01AA86FA0D1DC425ull,
       .c0_bits = 0x3E99999Au,
       .total_weight = 5,
       .trajectory = 0x53F1D9C82C26E009ull},
      {.name = "G2",
       .instance = rudy("four_nodes.rudy"),
       .algorithm = SbAlgorithm::ballistic,
       .c0 = 0.3f,
       .steps = 100,
       .seed = 2,
       .eval_every = 10,
       .optimum = 4,
       .best_step = {70, 60, 10, 70, 70, 70, 70, 10},
       .final_cut = {4, 4, 4, 4, 4, 4, 4, 4},
       .fingerprint = 0xABD93D3593EDC425ull,
       .c0_bits = 0x3E99999Au,
       .total_weight = 5,
       .trajectory = 0x70C8B29BC12705AEull},
      {.name = "G3",
       .instance = rudy("torus_4x6_pm1.rudy"),
       .algorithm = SbAlgorithm::discrete,
       .c0 = 0.0f,
       .steps = 100,
       .seed = 3,
       .eval_every = 0,
       .optimum = 24,
       .best_step = {40, 35, 21, 22, 19, 17, 37, 28},
       .final_cut = {24, 24, 24, 24, 24, 24, 24, 24},
       .fingerprint = 0x263996E44423941Bull,
       .c0_bits = 0x3E7D1123u,
       .total_weight = 16,
       .trajectory = 0x8DF7CB7BC5646913ull},
      {.name = "G4",
       .instance = rudy("torus_4x6_unit.rudy"),
       .algorithm = SbAlgorithm::ballistic,
       .c0 = 0.0f,
       .steps = 100,
       .seed = 3,
       .eval_every = 0,
       .optimum = 48,
       .best_step = {100, 100, 100, 100, 100, 100, 100, 100},
       .final_cut = {48, 48, 48, 48, 48, 48, 48, 48},
       .fingerprint = 0x59888B4D1012BD25ull,
       .c0_bits = 0x3E89DD8Du,
       .total_weight = 48,
       .trajectory = 0x7B4113BC797CE654ull},
      {.name = "G5",
       .instance = DenseSpinGlass{20, 20},
       .algorithm = SbAlgorithm::discrete,
       .c0 = 0.0f,
       .steps = 200,
       .seed = 4,
       .eval_every = 0,
       .optimum = 35,
       .best_step = {53, 35, 77, 47, 90, 84, 115, 75},
       .final_cut = {35, 34, 34, 34, 35, 35, 35, 35},
       .fingerprint = 0x17BF891DFB82BBA5ull,
       .c0_bits = 0x3DE54A89u,
       .total_weight = 10,
       .trajectory = 0x07E018CF0A497A77ull},
      {.name = "G6",
       .instance = DenseSpinGlass{24, 24},
       .algorithm = SbAlgorithm::ballistic,
       .c0 = 0.0f,
       .steps = 200,
       .seed = 5,
       .eval_every = 1,
       .optimum = 47,
       .best_step = {64, 76, 30, 95, 35, 103, 64, 70},
       .final_cut = {47, 47, 47, 47, 47, 47, 47, 47},
       .fingerprint = 0xA021B0A5174B4D25ull,
       .c0_bits = 0x3DD12914u,
       .total_weight = 10,
       .trajectory = 0x4F60D256CEE00FC2ull},
  };
}

// Instances built by the test (F05-C2, section 5.7).
// heavy_torus: the 48 edges of torus_4x6_unit.rudy with every weight multiplied by 10000.
Graph heavy_torus() {
  Graph g = rudy("torus_4x6_unit.rudy");
  for (sbmc::Edge& e : g.edges) {
    e.w *= 10000;
  }
  return g;
}

// dense_graph_40: n = 40 and, for 0 <= u < v < 40 in canonical order, the edge (u, v, w) with
// w = ((3u + 5v + uv) mod 7) - 3 when w != 0: 689 edges, |w| <= 3.
Graph dense_graph_40() {
  Graph g{40, {}};
  for (std::int32_t u = 0; u < 40; ++u) {
    for (std::int32_t v = u + 1; v < 40; ++v) {
      const std::int32_t w = ((3 * u + 5 * v + u * v) % 7) - 3;
      if (w != 0) {
        g.edges.push_back({u, v, w});
      }
    }
  }
  return g;
}

// dense_graph_40_heavy: dense_graph_40 with the weight of the edge (0, 1) raised from 2 to 200.
Graph dense_graph_40_heavy() {
  Graph g = dense_graph_40();
  g.edges.front().w = 200;
  return g;
}

}  // namespace

TEST_CASE("C1: test instances as specified") {
  const Graph dense40 = dense_graph_40();
  CHECK(dense40.edges.size() == 689u);
  REQUIRE(!dense40.edges.empty());
  CHECK(dense40.edges.front().u == 0);
  CHECK(dense40.edges.front().v == 1);
  CHECK(dense40.edges.front().w == 2);
  CHECK(dense_graph_40_heavy().edges.front().w == 200);
  CHECK(heavy_torus().edges.size() == 48u);
  CHECK(kLanes == sbmc::kSbOpenmpReplicasPerBlock);
}

TEST_CASE("C1: reference cases G1-G6 (T7)") {
  for (const ReferenceCase& c : reference_cases()) {
    for (const int threads : {1, 0}) {
      INFO(c.name << ", threads = " << threads);
      const SbResult result = sbmc::run_sb_openmp(
          c.instance, params(c.algorithm, 8, c.steps, c.c0, c.seed, c.eval_every), threads,
          c.steps);
      const auto n = static_cast<std::size_t>(sbmc::num_spins(c.instance));
      REQUIRE(result.per_replica.size() == 8u);
      REQUIRE(result.best_spins.size() == 8 * n);
      REQUIRE(result.final_x.size() == 8 * n);
      REQUIRE(result.final_y.size() == 8 * n);
      REQUIRE(result.trace_steps == c.steps);
      CHECK(result.threads == 1);  // R = 8 <= RB: one block

      if (kStrict) {
        INFO("fingerprint expected "
             << hex64(c.fingerprint) << ", got " << hex64(fingerprint(result))
             << "; trajectory expected " << hex64(c.trajectory) << ", got "
             << hex64(trajectory_fingerprint(result)) << "; c0 expected " << hex32(c.c0_bits)
             << ", got " << hex32(bits(result.c0)));
        CHECK(bits(result.c0) == c.c0_bits);
        CHECK(result.total_weight == c.total_weight);
        CHECK(fingerprint(result) == c.fingerprint);
        CHECK(trajectory_fingerprint(result) == c.trajectory);
        for (int r = 0; r < 8; ++r) {
          const auto& rep = result.per_replica[static_cast<std::size_t>(r)];
          const auto k = static_cast<std::size_t>(r);
          INFO("replica " << r << ": best_cut " << rep.best_cut << ", best_step " << rep.best_step
                          << ", final_cut " << rep.final_cut);
          CHECK(rep.best_cut == c.optimum);
          CHECK(rep.best_step == c.best_step[k]);
          CHECK(rep.final_cut == c.final_cut[k]);
        }
      }

      // Properties of every preset.
      bool optimum_reached = false;
      for (int r = 0; r < 8; ++r) {
        const auto& rep = result.per_replica[static_cast<std::size_t>(r)];
        INFO("replica " << r);
        optimum_reached = optimum_reached || rep.best_cut == c.optimum;
        CHECK(rep.best_cut <= c.optimum);
        CHECK(rep.best_cut == sbmc::cut_direct(c.instance, replica_spins(result, r)));
      }
      CHECK(optimum_reached);
      // With trace_steps = M the last recorded state is the final state.
      const std::size_t last = static_cast<std::size_t>(c.steps - 1) * 8 * n;
      const std::vector<float> last_x(result.trace_x.begin() + static_cast<std::ptrdiff_t>(last),
                                      result.trace_x.end());
      const std::vector<float> last_y(result.trace_y.begin() + static_cast<std::ptrdiff_t>(last),
                                      result.trace_y.end());
      CHECK(bit_differences(last_x, result.final_x) == 0u);
      CHECK(bit_differences(last_y, result.final_y) == 0u);
    }
  }
}

// GS-G1: G1 of the G-set, dSB, R = 16, M = 2000, dt = 0.5f, a0 = 1, automatic c0, seed 1,
// trace_steps = 100, two threads; constants of the case GS-G1 of test_sb_sequential.cpp.
TEST_CASE("C1: GS-G1, G1 of the G-set with dt = 0.5") {
  constexpr std::int64_t kBestKnown = 11624;
  const Instance g1 = g1_of_gset();
  SbParams p;
  p.algorithm = SbAlgorithm::discrete;
  p.replicas = 16;
  p.steps = 2000;
  p.dt = 0.5f;
  p.a0 = 1.0f;
  p.c0 = 0.0f;
  p.seed = 1;
  const SbResult result = sbmc::run_sb_openmp(g1, p, 2, 100);
  REQUIRE(result.per_replica.size() == 16u);
  REQUIRE(result.trace_x.size() == 100u * 16u * 800u);

  // Every preset.
  std::int64_t best = 0;
  for (int r = 0; r < 16; ++r) {
    const auto& rep = result.per_replica[static_cast<std::size_t>(r)];
    INFO("replica " << r << ": best_cut " << rep.best_cut);
    best = std::max(best, rep.best_cut);
    CHECK(rep.best_cut <= kBestKnown);
    CHECK(rep.best_cut == sbmc::cut_direct(g1, replica_spins(result, r)));
  }
  CHECK(best >= 11600);

  // Strict values.
  if (!kStrict) {
    MESSAGE("production preset: the exact values are checked only in the strict ones");
    return;
  }
  const std::vector<std::int64_t> best_cut = {11613, 11608, 11566, 11576, 11596, 11572,
                                              11587, 11603, 11620, 11576, 11559, 11613,
                                              11624, 11594, 11609, 11620};
  const std::vector<int> best_step = {1794, 1693, 1696, 1621, 1676, 1697, 1683, 1719,
                                      1692, 1637, 1645, 1736, 1696, 1513, 1769, 1667};
  for (std::size_t r = 0; r < 16; ++r) {
    const auto& rep = result.per_replica[r];
    INFO("replica " << r << ": best_cut " << rep.best_cut << ", best_step " << rep.best_step
                    << ", final_cut " << rep.final_cut);
    CHECK(rep.best_cut == best_cut[r]);
    CHECK(rep.best_step == best_step[r]);
    CHECK(rep.final_cut == best_cut[r]);
  }
  const std::uint64_t trajectory = trajectory_fingerprint(result);
  const std::uint64_t final_state = fingerprint(result);
  INFO("trajectory: expected 0x22C204C132E1ACAA, got " << hex64(trajectory)
                                                       << "; final state: expected "
                                                          "0x715609A2687E4759, got "
                                                       << hex64(final_state));
  CHECK(trajectory == 0x22C204C132E1ACAAull);
  CHECK(final_state == 0x715609A2687E4759ull);
}

// T7 against C0 itself: 9 instances x 4 configurations of the algorithm x 5 configurations of the
// replicas = 180 pairs of runs, seed 11, a0 = 1, trace_steps = 3. Format of J and accumulator that
// C1 uses (rule of sbmc/sb_openmp.hpp; int16 when L <= 32767):
//   data/gset/G1                 compressed  L = 67          int16
//   torus_4x6_pm1.rudy           compressed  L = 4           int16
//   heavy_torus                  compressed  L = 40000       int32
//   dense_graph_40               dense       L = 117         int16
//   dense_graph_40_heavy         compressed  L = 268         int16
//   weighted_isolated_13.rudy    compressed  L = 4294969294  int64
//   DenseSpinGlass{200, 7}       dense       L = 199         int16
//   Graph{1, {}}                 dense       L = 0           int16
//   DenseSpinGlass{1, 3}         dense       L = 0           int16
TEST_CASE("C1: equal to C0 field by field (strict)") {
  if (!kStrict) {
    MESSAGE("production preset: C1 and C0 are compared only in the strict ones");
    return;
  }
  const OmpSettingsGuard guard;
  omp_set_dynamic(0);
  const int initial_threads = guard.initial_max_threads();

  struct InstanceCase {
    std::string name;
    Instance instance;
    float dt;
    int steps;
    float c0;  // 0: auto
  };
  const std::vector<InstanceCase> instances = {
      {"data/gset/G1", g1_of_gset(), 0.5f, 60, 0.0f},
      {"torus_4x6_pm1.rudy", rudy("torus_4x6_pm1.rudy"), 1.25f, 100, 0.0f},
      {"heavy_torus", heavy_torus(), 0.5f, 100, 0.0f},
      {"dense_graph_40", dense_graph_40(), 1.25f, 100, 0.0f},
      {"dense_graph_40_heavy", dense_graph_40_heavy(), 1.25f, 100, 0.0f},
      {"weighted_isolated_13.rudy", rudy("weighted_isolated_13.rudy"), 0.5f, 100, 1e-9f},
      {"DenseSpinGlass{200, 7}", DenseSpinGlass{200, 7}, 1.25f, 60, 0.0f},
      {"Graph{1, {}}", Graph{1, {}}, 1.25f, 10, 0.5f},
      {"DenseSpinGlass{1, 3}", DenseSpinGlass{1, 3}, 1.25f, 10, 0.5f},
  };
  struct AlgorithmCase {
    const char* name;
    SbAlgorithm algorithm;
    int eval_every;
  };
  const std::vector<AlgorithmCase> algorithms = {{"dSB", SbAlgorithm::discrete, 0},
                                                 {"bSB eval_every 0", SbAlgorithm::ballistic, 0},
                                                 {"bSB eval_every 1", SbAlgorithm::ballistic, 1},
                                                 {"bSB eval_every 7", SbAlgorithm::ballistic, 7}};
  struct ReplicaCase {
    int replicas;
    std::int64_t first_replica;
    int threads;
  };
  const std::vector<ReplicaCase> splits = {
      {1, 0, 1}, {5, 7, 2}, {17, 0, 3}, {33, 7, 0}, {64, 0, initial_threads}};

  int pairs = 0;
  for (const InstanceCase& ic : instances) {
    for (const AlgorithmCase& ac : algorithms) {
      for (const ReplicaCase& rc : splits) {
        SbParams p = params(ac.algorithm, rc.replicas, ic.steps, ic.c0, 11, ac.eval_every);
        p.dt = ic.dt;
        p.first_replica = rc.first_replica;
        INFO(ic.name << ", " << ac.name << ", R = " << rc.replicas
                     << ", first_replica = " << rc.first_replica << ", threads = " << rc.threads);
        const SbResult c0 = sbmc::run_sb_sequential(ic.instance, p, 3);
        const SbResult c1 = sbmc::run_sb_openmp(ic.instance, p, rc.threads, 3);
        const std::string differ = field_differences(c0, c1);
        INFO("fields that differ: " << (differ.empty() ? std::string("none") : differ));
        CHECK(differ.empty());
        CHECK(c1.threads == expected_team(rc.threads, rc.replicas));
        ++pairs;
      }
    }
  }
  CHECK(pairs == 180);
  CHECK(omp_get_max_threads() == initial_threads);
}

// The integer accumulators at their limits (added in F05-R2, not in the table of F05-C2): spin 0
// joined to spin 1 with weight w and to spin 2 with weight 1 gives L = w + 1, so w = 32766, 32767,
// 2^31 - 2 and 2^31 - 1 give L = 32767 (int16), 32768 (int32), 2^31 - 1 (int32) and 2^31 (int64).
// The field of spin 0 is +L when s_1 = s_2 = -1, which a narrower accumulator cannot hold; with
// R = RB + 1 every algorithm evaluates such configurations from step 0. W = L = the maximum cut.
// In every preset the final cut of every replica must be the cut of the signs of its final x: a
// final configuration with s_1 = s_2 = -1 and an overflowed field would give another value.
TEST_CASE("C1: integer accumulators at their limits") {
  struct Limit {
    const char* name;
    std::int32_t w;
  };
  const std::vector<Limit> limits = {{"L = 32767, int16", 32766},
                                     {"L = 32768, int32", 32767},
                                     {"L = 2^31 - 1, int32", 2147483646},
                                     {"L = 2^31, int64", 2147483647}};
  for (const Limit& limit : limits) {
    const Graph g{3, {{0, 1, limit.w}, {0, 2, 1}}};
    const std::int64_t norm = std::int64_t{limit.w} + 1;
    const auto c0 = static_cast<float>(0.5 / static_cast<double>(norm));
    for (const SbAlgorithm algorithm : {SbAlgorithm::discrete, SbAlgorithm::ballistic}) {
      const SbParams p = params(algorithm, kLanes + 1, 50, c0, 13, 1);
      INFO(limit.name << (algorithm == SbAlgorithm::discrete ? ", dSB" : ", bSB eval_every 1"));
      SbResult c1;
      REQUIRE_NOTHROW(c1 = sbmc::run_sb_openmp(g, p, 2, 3));  // final check with cut_direct
      CHECK(c1.total_weight == norm);
      CHECK(c1.threads == expected_team(2, p.replicas));
      REQUIRE(c1.per_replica.size() == static_cast<std::size_t>(p.replicas));
      for (int r = 0; r < p.replicas; ++r) {
        const auto& replica = c1.per_replica[static_cast<std::size_t>(r)];
        CHECK(replica.best_cut <= norm);
        CHECK(sbmc::cut_direct(g, replica_spins(c1, r)) == replica.best_cut);
        std::vector<Spin> final_signs(3);
        for (std::size_t i = 0; i < final_signs.size(); ++i) {
          final_signs[i] = sbmc::sb_sign(c1.final_x[static_cast<std::size_t>(r) * 3 + i]);
        }
        CHECK(sbmc::cut_direct(g, final_signs) == replica.final_cut);
      }
      if (kStrict) {
        const std::string differ = field_differences(sbmc::run_sb_sequential(g, p, 3), c1);
        INFO("fields that differ: " << (differ.empty() ? std::string("none") : differ));
        CHECK(differ.empty());
      }
    }
  }
}

TEST_CASE("C1: the result does not depend on the number of threads") {
  const OmpSettingsGuard guard;
  omp_set_dynamic(0);
  const int initial_threads = guard.initial_max_threads();
  struct Case {
    std::string name;
    Instance instance;
    SbParams p;
    int trace_steps;
  };
  SbParams g1_params = params(SbAlgorithm::ballistic, 64, 60, 0.0f, 11, 7);
  g1_params.dt = 0.5f;
  const std::vector<Case> cases = {
      {"data/gset/G1, bSB, eval_every 7, R = 64", g1_of_gset(), g1_params, 2},
      {"dense_graph_40, dSB, R = 33", dense_graph_40(),
       params(SbAlgorithm::discrete, 33, 100, 0.0f, 11, 0), 0},
      {"DenseSpinGlass{200, 7}, bSB, eval_every 1, R = 17", DenseSpinGlass{200, 7},
       params(SbAlgorithm::ballistic, 17, 60, 0.0f, 11, 1), 0},
  };
  const std::vector<int> thread_counts = {1, 2, 3, 0, initial_threads};
  for (const Case& c : cases) {
    std::vector<SbResult> results;
    for (const int threads : thread_counts) {
      INFO(c.name << ", threads = " << threads);
      results.push_back(sbmc::run_sb_openmp(c.instance, c.p, threads, c.trace_steps));
      CHECK(results.back().threads == expected_team(threads, c.p.replicas));
      // The library changes no OpenMP setting.
      CHECK(omp_get_max_threads() == initial_threads);
      CHECK(omp_get_dynamic() == 0);
    }
    for (std::size_t k = 1; k < results.size(); ++k) {
      const std::string differ = field_differences(results[k], results[0]);
      INFO(c.name << ", threads = " << thread_counts[k] << " against threads = 1: fields that "
                  << "differ: " << (differ.empty() ? std::string("none") : differ));
      CHECK(differ.empty());
    }
  }
  omp_set_dynamic(guard.initial_dynamic());
  omp_set_num_threads(initial_threads);
  CHECK(omp_get_max_threads() == initial_threads);
  CHECK(omp_get_dynamic() == guard.initial_dynamic());
}

TEST_CASE("C1: splitting the replicas with first_replica gives the same replicas") {
  SbParams g1_params = params(SbAlgorithm::ballistic, 33, 60, 0.0f, 11, 7);
  g1_params.dt = 0.5f;
  const std::vector<std::pair<std::string, std::pair<Instance, SbParams>>> cases = {
      {"data/gset/G1, bSB, eval_every 7", {g1_of_gset(), g1_params}},
      {"dense_graph_40, dSB",
       {dense_graph_40(), params(SbAlgorithm::discrete, 33, 100, 0.0f, 11, 0)}},
  };
  for (const auto& [name, setup] : cases) {
    INFO(name);
    const auto& [instance, base] = setup;
    SbParams all = base;
    all.replicas = 33;
    all.first_replica = 7;
    SbParams first = base;
    first.replicas = 10;
    first.first_replica = 7;
    SbParams second = base;
    second.replicas = 23;
    second.first_replica = 17;
    const SbResult whole = sbmc::run_sb_openmp(instance, all, 3);
    const SbResult a = sbmc::run_sb_openmp(instance, first, 3);
    const SbResult b = sbmc::run_sb_openmp(instance, second, 3);
    CHECK(b.first_replica == 17);
    SbResult joined = a;
    joined.per_replica.insert(joined.per_replica.end(), b.per_replica.begin(), b.per_replica.end());
    joined.best_spins.insert(joined.best_spins.end(), b.best_spins.begin(), b.best_spins.end());
    joined.final_x.insert(joined.final_x.end(), b.final_x.begin(), b.final_x.end());
    joined.final_y.insert(joined.final_y.end(), b.final_y.begin(), b.final_y.end());
    CHECK(same_replica_results(joined, whole));
    CHECK(bit_differences(joined.final_x, whole.final_x) == 0u);
    CHECK(bit_differences(joined.final_y, whole.final_y) == 0u);
  }
}

TEST_CASE("C1: the c0 actually used") {
  const Instance inst = rudy("torus_4x6_pm1.rudy");
  const float automatic = sbmc::auto_c0(inst);
  const SbResult explicit_c0 =
      sbmc::run_sb_openmp(inst, params(SbAlgorithm::discrete, 2, 10, 0.4f, 1, 0));
  CHECK(bits(explicit_c0.c0) == bits(0.4f));
  for (const float zero : {0.0f, -0.0f}) {
    const SbResult result =
        sbmc::run_sb_openmp(inst, params(SbAlgorithm::ballistic, 2, 10, zero, 1, 0));
    INFO("c0 = " << std::string(std::signbit(zero) ? "-0.0f" : "+0.0f"));
    CHECK(bits(result.c0) == bits(automatic));
  }
}

TEST_CASE("C1: instances with a single spin") {
  const std::vector<Instance> instances = {Graph{1, {}}, DenseSpinGlass{1, 3}};
  for (const Instance& inst : instances) {
    for (const SbAlgorithm algorithm : {SbAlgorithm::discrete, SbAlgorithm::ballistic}) {
      const bool discrete = algorithm == SbAlgorithm::discrete;
      INFO(std::string(inst.index() == 0 ? "Graph" : "DenseSpinGlass")
           << ", " << std::string(discrete ? "dSB" : "bSB"));
      const SbResult result = sbmc::run_sb_openmp(inst, params(algorithm, 2, 10, 0.5f, 9, 0));
      REQUIRE(result.per_replica.size() == 2u);
      for (const auto& rep : result.per_replica) {
        CHECK(rep.best_energy == 0);
        CHECK(rep.best_cut == 0);
        CHECK(rep.final_cut == 0);
        CHECK(rep.best_step == (discrete ? 0 : 10));
      }
    }
  }
}

TEST_CASE("C1: trace_steps = 0 records nothing; C0 reports one thread; RB") {
  const SbResult result = sbmc::run_sb_openmp(
      rudy("four_nodes.rudy"), params(SbAlgorithm::discrete, 3, 20, 0.3f, 1, 0), 0, 0);
  CHECK(result.trace_steps == 0);
  CHECK(result.trace_x.empty());
  CHECK(result.trace_y.empty());
  CHECK(result.final_x.size() == 12u);
  CHECK(result.final_y.size() == 12u);

  const SbResult sequential = sbmc::run_sb_sequential(
      rudy("four_nodes.rudy"), params(SbAlgorithm::discrete, 3, 20, 0.3f, 1, 0));
  CHECK(sequential.threads == 1);

  CHECK((kLanes == 8 || kLanes == 16 || kLanes == 32));
}

TEST_CASE("C1: invalid arguments throw InvalidArgument") {
  const Instance four = rudy("four_nodes.rudy");
  const SbParams good = params(SbAlgorithm::discrete, 2, 10, 0.3f, 1, 0);
  CHECK_NOTHROW(sbmc::run_sb_openmp(four, good, 0, 10));

  // threads < 0, with the prefix of the function.
  std::string message;
  try {
    static_cast<void>(sbmc::run_sb_openmp(four, good, -1));
  } catch (const InvalidArgument& e) {
    message = e.what();
  }
  INFO("threads = -1: '" << message << "'");
  CHECK(message.starts_with("run_sb_openmp: "));

  // The cases of "C0: invalid arguments throw InvalidArgument".
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  struct Case {
    std::string name;
    SbParams p;
    int trace_steps;
  };
  std::vector<Case> cases;
  const auto with = [&](const std::string& name, auto change, int trace_steps = 0) {
    SbParams p = good;
    change(p);
    cases.push_back({name, p, trace_steps});
  };
  with("replicas 0", [](SbParams& p) { p.replicas = 0; });
  with("steps 0", [](SbParams& p) { p.steps = 0; });
  with("dt 0", [](SbParams& p) { p.dt = 0.0f; });
  with("dt -1", [](SbParams& p) { p.dt = -1.0f; });
  with("dt NaN", [&](SbParams& p) { p.dt = nan; });
  with("dt +inf", [&](SbParams& p) { p.dt = inf; });
  with("a0 0", [](SbParams& p) { p.a0 = 0.0f; });
  with("c0 -1", [](SbParams& p) { p.c0 = -1.0f; });
  with("c0 NaN", [&](SbParams& p) { p.c0 = nan; });
  with("c0 +inf", [&](SbParams& p) { p.c0 = inf; });
  with("eval_every -1", [](SbParams& p) { p.eval_every = -1; });
  with("first_replica -1", [](SbParams& p) { p.first_replica = -1; });
  with("first_replica 2^32 - 2 with 3 replicas", [](SbParams& p) {
    p.first_replica = (std::int64_t{1} << 32) - 2;
    p.replicas = 3;
  });
  with("trace_steps -1", [](SbParams&) {}, -1);
  with("trace_steps M + 1", [](SbParams&) {}, 11);
  with("algorithm 2", [](SbParams& p) { p.algorithm = static_cast<SbAlgorithm>(2); });
  for (const Case& c : cases) {
    INFO(c.name);
    CHECK_THROWS_AS(sbmc::run_sb_openmp(four, c.p, 0, c.trace_steps), InvalidArgument);
  }
  std::string replicas_message;
  try {
    SbParams p = good;
    p.replicas = 0;
    static_cast<void>(sbmc::run_sb_openmp(four, p));
  } catch (const InvalidArgument& e) {
    replicas_message = e.what();
  }
  INFO("replicas 0: '" << replicas_message << "'");
  CHECK(replicas_message.starts_with("run_sb_openmp: "));

  CHECK_THROWS_AS(sbmc::run_sb_openmp(Graph{4, {{0, 2, 1}, {0, 1, 1}}}, good),
                  InvalidArgument);  // edges out of order
  // n is checked before total_weight, which would take O(n^2) time here.
  CHECK_THROWS_AS(sbmc::run_sb_openmp(DenseSpinGlass{32769, 1}, good), InvalidArgument);
  SbParams automatic = good;
  automatic.c0 = 0.0f;
  CHECK_THROWS_AS(sbmc::run_sb_openmp(Graph{2, {{0, 1, 1}}}, automatic), InvalidArgument);

  // The last valid global replica index: no exception, and the same result as C0 in the strict
  // presets.
  SbParams last = good;
  last.first_replica = (std::int64_t{1} << 32) - 2;
  last.replicas = 2;
  SbResult c1;
  CHECK_NOTHROW(c1 = sbmc::run_sb_openmp(four, last, 0, 2));
  if (kStrict) {
    const SbResult c0 = sbmc::run_sb_sequential(four, last, 2);
    const std::string differ = field_differences(c0, c1);
    INFO("fields that differ: " << (differ.empty() ? std::string("none") : differ));
    CHECK(differ.empty());
  }
}
