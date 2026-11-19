// C2, simulated annealing with OpenMP over blocks of replicas (sbmc/sa_openmp.hpp): the constants
// SA-1 to SA-9, C2 against C0 field by field, independence from the number of threads and from
// the split of the replicas, high global indices, other properties, preconditions.
//
// SA-1 to SA-7 are the constants of test_sa_sequential.cpp; SA-8 and SA-9 come from F06-C2,
// section 5.7 (an emulator of cldArchitect independent of the code of the project) and are
// checked here with run_sa_sequential and with run_sa_openmp. SA has no floating-point operation
// outside sa_beta and sa_threshold, so C2 must give every field of C0 bit for bit in every preset,
// strict or not. The helper functions are copies of those of test_sa_sequential.cpp and
// test_sb_openmp.cpp.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <omp.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/rudy.hpp>
#include <sbmc/sa.hpp>
#include <sbmc/sa_openmp.hpp>
#include <sbmc/sa_sequential.hpp>

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
using sbmc::SaParams;
using sbmc::SaResult;
using sbmc::Spin;

constexpr int kLanes = sbmc::kSaOpenmpReplicasPerBlock;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

Graph rudy(const char* name) {
  return sbmc::read_rudy(std::string(SBMC_TEST_DATA_DIR) + "/" + name);
}

Graph g1_of_gset() {
  return sbmc::read_rudy(std::string(SBMC_DATA_DIR) + "/gset/G1");
}

SaParams params(int replicas, int sweeps, double beta0, double beta1, std::uint64_t seed,
                std::int64_t first_replica = 0) {
  SaParams p;
  p.replicas = replicas;
  p.sweeps = sweeps;
  p.beta0 = beta0;
  p.beta1 = beta1;
  p.seed = seed;
  p.first_replica = first_replica;
  return p;
}

// FNV-1a 64 over the bytes (uint8_t) of best_spins, then of final_spins (+1 is 0x01, -1 is 0xFF).
std::uint64_t fingerprint(const SaResult& result) {
  std::uint64_t hash = 0xCBF29CE484222325ULL;
  for (const std::vector<Spin>* spins : {&result.best_spins, &result.final_spins}) {
    for (const Spin s : *spins) {
      hash ^= static_cast<std::uint8_t>(s);
      hash *= 0x100000001B3ULL;
    }
  }
  return hash;
}

bool same_replica(const sbmc::SaReplicaResult& x, const sbmc::SaReplicaResult& y) {
  return x.best_energy == y.best_energy && x.best_cut == y.best_cut &&
         x.best_sweep == y.best_sweep && x.final_cut == y.final_cut && x.accepted == y.accepted;
}

bool same_per_replica(const std::vector<sbmc::SaReplicaResult>& a,
                      const std::vector<sbmc::SaReplicaResult>& b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), same_replica);
}

// The fields that differ between two results, threads excluded; empty when they are equal field
// by field (n, replicas, sweeps, first_replica, the bits of beta0 and beta1, total_weight, the five
// fields of every per_replica, best_spins, final_spins).
std::string field_differences(const SaResult& a, const SaResult& b) {
  std::string differ;
  const auto add = [&differ](bool different, const char* field) {
    if (different) {
      differ += differ.empty() ? field : std::string(", ") + field;
    }
  };
  add(a.n != b.n, "n");
  add(a.replicas != b.replicas, "replicas");
  add(a.sweeps != b.sweeps, "sweeps");
  add(a.first_replica != b.first_replica, "first_replica");
  add(std::bit_cast<std::uint64_t>(a.beta0) != std::bit_cast<std::uint64_t>(b.beta0), "beta0");
  add(std::bit_cast<std::uint64_t>(a.beta1) != std::bit_cast<std::uint64_t>(b.beta1), "beta1");
  add(a.total_weight != b.total_weight, "total_weight");
  add(!same_per_replica(a.per_replica, b.per_replica), "per_replica");
  add(a.best_spins != b.best_spins, "best_spins");
  add(a.final_spins != b.final_spins, "final_spins");
  return differ;
}

std::string shown(const std::string& differ) {
  return differ.empty() ? std::string("none") : differ;
}

// The team of C2: min(T_req, B), with T_req = threads or omp_get_max_threads() for threads = 0.
int expected_team(int threads, int replicas) {
  const int requested = threads > 0 ? threads : omp_get_max_threads();
  return std::min(requested, (replicas + kLanes - 1) / kLanes);
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

// Instances built by the test (F05-C2, section 5.7, and F06-C2, section 5.6).
// heavy_torus: the 48 edges of torus_4x6_unit.rudy with every weight multiplied by 10000.
Graph scaled_torus(std::int32_t factor) {
  Graph g = rudy("torus_4x6_unit.rudy");
  for (sbmc::Edge& e : g.edges) {
    e.w *= factor;
  }
  return g;
}

Graph heavy_torus() {
  return scaled_torus(10000);
}

// huge_torus: the 48 edges of torus_4x6_unit.rudy with every weight multiplied by 500000.
Graph huge_torus() {
  return scaled_torus(500000);
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

struct ReferenceCase {
  const char* name;
  Instance instance;
  SaParams params;
  std::int64_t total_weight;
  std::vector<std::int64_t> best_cut;
  std::vector<int> best_sweep;
  std::vector<std::int64_t> final_cut;
  std::vector<std::int64_t> accepted;
  std::uint64_t fingerprint;
};

// SA-1 to SA-7: the constants of test_sa_sequential.cpp (F06-C1, section 5.8).
std::vector<ReferenceCase> reference_cases() {
  std::vector<ReferenceCase> cases;
  cases.push_back({"SA-1 four_nodes.rudy",
                   rudy("four_nodes.rudy"),
                   params(8, 10, 0.1, 3.0, 2),
                   5,
                   {4, 4, 4, 4, 4, 4, 4, 4},
                   {6, 0, 2, 1, 0, 3, 0, 1},
                   {4, 4, 4, 4, 4, 4, 4, 4},
                   {24, 20, 17, 15, 18, 24, 20, 22},
                   0x981cc2c5c4d485e5ULL});
  cases.push_back({"SA-2 torus_4x6_pm1.rudy",
                   rudy("torus_4x6_pm1.rudy"),
                   params(8, 50, 0.1, 3.0, 3),
                   16,
                   {24, 24, 24, 24, 24, 24, 24, 24},
                   {31, 23, 30, 20, 21, 25, 26, 25},
                   {24, 24, 24, 24, 24, 24, 24, 24},
                   {520, 489, 505, 514, 487, 473, 471, 519},
                   0xbc8f2be192279e69ULL});
  cases.push_back({"SA-3 DenseSpinGlass{20, 20}",
                   DenseSpinGlass{20, 20},
                   params(8, 100, 0.1, 3.0, 4),
                   10,
                   {35, 35, 35, 35, 35, 35, 35, 35},
                   {18, 38, 27, 33, 25, 18, 20, 38},
                   {35, 35, 35, 35, 34, 35, 34, 35},
                   {369, 361, 357, 355, 399, 387, 378, 387},
                   0xebb72b8d3bacd7b1ULL});
  cases.push_back({"SA-4 weighted_isolated_13.rudy",
                   rudy("weighted_isolated_13.rudy"),
                   params(8, 20, 1e-10, 1e-9, 5),
                   2147483669,
                   {6442452014, 6442452013, 6442452014, 6442452014, 6442452014, 6442452014,
                    6442451974, 6442452013},
                   {3, 9, 15, 15, 7, 7, 15, 2},
                   {6442451973, 6442450953, 4294967294, 6442450975, 6442451973, 6442451904,
                    6442450965, 6442451916},
                   {152, 167, 178, 167, 166, 172, 175, 160},
                   0xc675acc36d13b2c7ULL});
  cases.push_back({"SA-5 data/gset/G1",
                   g1_of_gset(),
                   params(4, 20, 0.1, 3.0, 1),
                   19176,
                   {11532, 11465, 11498, 11517},
                   {18, 18, 20, 18},
                   {11532, 11465, 11498, 11517},
                   {2285, 2348, 2373, 2422},
                   0xeea4b13999139ab9ULL});
  cases.push_back({"SA-6 four_nodes.rudy, one sweep",
                   rudy("four_nodes.rudy"),
                   params(4, 1, 0.5, 0.5, 7),
                   5,
                   {4, 3, 3, 3},
                   {1, 0, 0, 0},
                   {3, 2, 3, 3},
                   {4, 2, 2, 2},
                   0x245c54e6779d5ff9ULL});
  cases.push_back({"SA-7 sparse_24_unit.rudy, first_replica 5",
                   rudy("sparse_24_unit.rudy"),
                   params(3, 30, 0.2, 2.0, 9, 5),
                   48,
                   {40, 40, 39},
                   {21, 23, 14},
                   {40, 40, 36},
                   {226, 228, 245},
                   0x231789d5863d723dULL});
  return cases;
}

// SA-8 and SA-9 (F06-C2, section 5.7): sums over the 70 replicas, the fingerprint, replicas 0 to
// 2 and 69 as (best_cut, best_sweep, final_cut, accepted).
struct SummaryCase {
  const char* name;
  Instance instance;
  SaParams params;
  std::int64_t total_weight;  // -1: from total_weight
  std::int64_t sum_best_cut;
  std::int64_t max_best_cut;
  std::int64_t min_best_cut;
  std::int64_t sum_best_sweep;
  std::int64_t sum_final_cut;
  std::int64_t sum_accepted;
  std::uint64_t fingerprint;
  std::vector<std::pair<int, std::vector<std::int64_t>>> replicas;
};

std::vector<SummaryCase> summary_cases() {
  std::vector<SummaryCase> cases;
  cases.push_back({"SA-8 DenseSpinGlass{300, 9}",
                   DenseSpinGlass{300, 9},
                   params(70, 30, 0.01, 3.0, 21, 100),
                   -1,
                   140062,
                   2071,
                   1913,
                   1722,
                   140056,
                   163679,
                   0x8fa18e99ef5b3d91ULL,
                   {{0, {2066, 30, 2066, 2310}},
                    {1, {2019, 26, 2019, 2333}},
                    {2, {2040, 24, 2040, 2402}},
                    {69, {1937, 21, 1937, 2373}}}});
  cases.push_back({"SA-9 data/gset/G1, the last 70 global indices",
                   g1_of_gset(),
                   params(70, 30, 0.1, 3.0, 22, 4294967226),
                   19176,
                   807087,
                   11580,
                   11476,
                   1860,
                   807082,
                   227005,
                   0x7d2fc5b7100267a9ULL,
                   {{0, {11519, 27, 11519, 3359}},
                    {1, {11514, 22, 11514, 3209}},
                    {2, {11530, 28, 11530, 3416}},
                    {69, {11520, 23, 11520, 3168}}}});
  return cases;
}

void check_summary(const SummaryCase& c, const SaResult& result) {
  const std::int64_t total = c.total_weight >= 0 ? c.total_weight : sbmc::total_weight(c.instance);
  CHECK(result.total_weight == total);
  REQUIRE(result.per_replica.size() == static_cast<std::size_t>(c.params.replicas));
  std::int64_t sum_best = 0;
  std::int64_t sum_sweep = 0;
  std::int64_t sum_final = 0;
  std::int64_t sum_accepted = 0;
  std::int64_t max_best = std::numeric_limits<std::int64_t>::min();
  std::int64_t min_best = std::numeric_limits<std::int64_t>::max();
  for (const sbmc::SaReplicaResult& out : result.per_replica) {
    sum_best += out.best_cut;
    sum_sweep += out.best_sweep;
    sum_final += out.final_cut;
    sum_accepted += out.accepted;
    max_best = std::max(max_best, out.best_cut);
    min_best = std::min(min_best, out.best_cut);
    CHECK(out.best_energy == total - 2 * out.best_cut);
  }
  CHECK(sum_best == c.sum_best_cut);
  CHECK(max_best == c.max_best_cut);
  CHECK(min_best == c.min_best_cut);
  CHECK(sum_sweep == c.sum_best_sweep);
  CHECK(sum_final == c.sum_final_cut);
  CHECK(sum_accepted == c.sum_accepted);
  CHECK(fingerprint(result) == c.fingerprint);
  for (const auto& [r, expected] : c.replicas) {
    INFO("replica " << r);
    const sbmc::SaReplicaResult& out = result.per_replica[static_cast<std::size_t>(r)];
    CHECK(out.best_cut == expected[0]);
    CHECK(out.best_sweep == expected[1]);
    CHECK(out.final_cut == expected[2]);
    CHECK(out.accepted == expected[3]);
  }
}

template <typename Function>
std::string message_of(Function&& function) {
  try {
    function();
  } catch (const InvalidArgument& e) {
    return e.what();
  }
  return "(no exception)";
}

}  // namespace

TEST_CASE("C2: reference cases SA-1 to SA-9") {
  for (const ReferenceCase& c : reference_cases()) {
    for (const int threads : {1, 0}) {
      INFO(c.name << ", threads = " << threads);
      const SaResult result = sbmc::run_sa_openmp(c.instance, c.params, threads);
      const auto replicas = static_cast<std::size_t>(c.params.replicas);
      CHECK(result.n == sbmc::num_spins(c.instance));
      CHECK(result.replicas == c.params.replicas);
      CHECK(result.sweeps == c.params.sweeps);
      CHECK(result.first_replica == c.params.first_replica);
      CHECK(std::bit_cast<std::uint64_t>(result.beta0) ==
            std::bit_cast<std::uint64_t>(c.params.beta0));
      CHECK(std::bit_cast<std::uint64_t>(result.beta1) ==
            std::bit_cast<std::uint64_t>(c.params.beta1));
      CHECK(result.total_weight == c.total_weight);
      CHECK(result.threads == expected_team(threads, c.params.replicas));
      REQUIRE(result.per_replica.size() == replicas);
      REQUIRE(result.best_spins.size() == replicas * static_cast<std::size_t>(result.n));
      REQUIRE(result.final_spins.size() == replicas * static_cast<std::size_t>(result.n));
      for (std::size_t r = 0; r < replicas; ++r) {
        INFO("replica " << r);
        const sbmc::SaReplicaResult& out = result.per_replica[r];
        CHECK(out.best_cut == c.best_cut[r]);
        CHECK(out.best_energy == c.total_weight - 2 * c.best_cut[r]);
        CHECK(out.best_sweep == c.best_sweep[r]);
        CHECK(out.final_cut == c.final_cut[r]);
        CHECK(out.accepted == c.accepted[r]);
      }
      CHECK(fingerprint(result) == c.fingerprint);
    }
  }
  for (const SummaryCase& c : summary_cases()) {
    {
      INFO(c.name << ", run_sa_sequential");
      const SaResult result = sbmc::run_sa_sequential(c.instance, c.params);
      CHECK(result.threads == 1);
      check_summary(c, result);
    }
    for (const int threads : {1, 0}) {
      INFO(c.name << ", run_sa_openmp, threads = " << threads);
      const SaResult result = sbmc::run_sa_openmp(c.instance, c.params, threads);
      CHECK(result.threads == expected_team(threads, c.params.replicas));
      CHECK(result.first_replica == c.params.first_replica);
      check_summary(c, result);
    }
  }
}

// C2 against C0: 10 instances x 5 configurations of the replicas = 50 pairs. Format of J,
// accumulator and path of the thresholds that C2 uses (rule of sbmc/sa_openmp.hpp; int16 when
// L <= 32767; table when L <= L_tab = 4096):
//   data/gset/G1                 compressed  L = 67          int16  table
//   torus_4x6_pm1.rudy           compressed  L = 4           int16  table
//   heavy_torus                  compressed  L = 40000       int32  direct
//   huge_torus                   compressed  L = 2000000     int32  direct
//   weighted_isolated_13.rudy    compressed  L = 4294969294  int64  direct
//   dense_graph_40               dense       L = 117         int16  table
//   dense_graph_40_heavy         compressed  L = 268         int16  table
//   DenseSpinGlass{200, 7}       dense       L = 199         int16  table
//   Graph{1, {}}                 dense       L = 0           int16  table
//   DenseSpinGlass{1, 3}         dense       L = 0           int16  table
// The case after this one adds the fifth combination, int16 with direct calls (4096 < L <= 32767).
TEST_CASE("C2: equal to C0 field by field") {
  const OmpSettingsGuard guard;
  omp_set_dynamic(0);
  const int initial_threads = guard.initial_max_threads();

  struct InstanceCase {
    std::string name;
    Instance instance;
    int sweeps;
    double beta0;
    double beta1;
  };
  const std::vector<InstanceCase> instances = {
      {"data/gset/G1", g1_of_gset(), 20, 0.1, 3.0},
      {"torus_4x6_pm1.rudy", rudy("torus_4x6_pm1.rudy"), 50, 0.1, 3.0},
      {"heavy_torus", heavy_torus(), 50, 1e-5, 1e-3},
      {"huge_torus", huge_torus(), 50, 1e-7, 1e-5},
      {"weighted_isolated_13.rudy", rudy("weighted_isolated_13.rudy"), 30, 1e-10, 1e-9},
      {"dense_graph_40", dense_graph_40(), 50, 0.05, 3.0},
      {"dense_graph_40_heavy", dense_graph_40_heavy(), 50, 0.05, 3.0},
      {"DenseSpinGlass{200, 7}", DenseSpinGlass{200, 7}, 30, 0.01, 3.0},
      {"Graph{1, {}}", Graph{1, {}}, 5, 0.5, 1.0},
      {"DenseSpinGlass{1, 3}", DenseSpinGlass{1, 3}, 5, 0.5, 1.0},
  };
  struct ReplicaCase {
    int replicas;
    std::int64_t first_replica;
    int threads;
  };
  const std::vector<ReplicaCase> replica_cases = {
      {1, 0, 1}, {5, 7, 2}, {17, 0, 3}, {kLanes + 1, 7, 0}, {2 * kLanes + 5, 0, initial_threads},
  };
  int pairs = 0;
  for (const InstanceCase& ic : instances) {
    for (const ReplicaCase& rc : replica_cases) {
      INFO(ic.name << ", R = " << rc.replicas << ", first_replica = " << rc.first_replica
                   << ", threads = " << rc.threads);
      const SaParams p = params(rc.replicas, ic.sweeps, ic.beta0, ic.beta1, 11, rc.first_replica);
      const SaResult c0 = sbmc::run_sa_sequential(ic.instance, p);
      const SaResult c2 = sbmc::run_sa_openmp(ic.instance, p, rc.threads);
      const std::string differ = field_differences(c0, c2);
      INFO("fields that differ: " << shown(differ));
      CHECK(differ.empty());
      CHECK(c2.threads == expected_team(rc.threads, rc.replicas));
      ++pairs;
    }
  }
  CHECK(pairs == 50);
}

// Added in F06-R2: the instantiation int16 with direct calls, which no instance of the case above
// reaches (torus_4x6_unit.rudy with every weight multiplied by 2000: compressed, L = 8000).
TEST_CASE("C2: int16 accumulator with direct thresholds, equal to C0") {
  const Graph mid_torus = scaled_torus(2000);
  for (const int replicas : {3, kLanes + 1}) {
    INFO("R = " << replicas);
    const SaParams p = params(replicas, 50, 1e-4, 1e-2, 11, 3);
    const std::string differ = field_differences(sbmc::run_sa_sequential(mid_torus, p),
                                                 sbmc::run_sa_openmp(mid_torus, p, 2));
    INFO("fields that differ: " << shown(differ));
    CHECK(differ.empty());
  }
}

TEST_CASE("C2: the result does not depend on the number of threads") {
  const OmpSettingsGuard guard;
  omp_set_dynamic(0);
  const int initial_threads = guard.initial_max_threads();
  struct Case {
    std::string name;
    Instance instance;
    SaParams p;
  };
  const std::vector<Case> cases = {
      {"data/gset/G1", g1_of_gset(), params(2 * kLanes + 5, 30, 0.1, 3.0, 5)},
      {"dense_graph_40", dense_graph_40(), params(kLanes + 1, 50, 0.05, 3.0, 6)},
      {"DenseSpinGlass{200, 7}", DenseSpinGlass{200, 7}, params(17, 30, 0.01, 3.0, 7)},
  };
  const std::vector<int> thread_counts = {1, 2, 3, 0, initial_threads};
  for (const Case& c : cases) {
    std::vector<SaResult> results;
    for (const int threads : thread_counts) {
      INFO(c.name << ", threads = " << threads);
      results.push_back(sbmc::run_sa_openmp(c.instance, c.p, threads));
      CHECK(results.back().threads == expected_team(threads, c.p.replicas));
      // The library changes no OpenMP setting.
      CHECK(omp_get_max_threads() == initial_threads);
      CHECK(omp_get_dynamic() == 0);
    }
    for (std::size_t k = 1; k < results.size(); ++k) {
      const std::string differ = field_differences(results[k], results[0]);
      INFO(c.name << ", threads = " << thread_counts[k]
                  << " against threads = 1: fields that differ: " << shown(differ));
      CHECK(differ.empty());
    }
  }
  omp_set_dynamic(guard.initial_dynamic());
  omp_set_num_threads(initial_threads);
  CHECK(omp_get_max_threads() == initial_threads);
  CHECK(omp_get_dynamic() == guard.initial_dynamic());
}

TEST_CASE("C2: splitting the replicas with first_replica gives the same replicas") {
  struct Case {
    std::string name;
    Instance instance;
    SaParams base;
  };
  const std::vector<Case> cases = {
      {"data/gset/G1", g1_of_gset(), params(1, 20, 0.1, 3.0, 3)},
      {"dense_graph_40", dense_graph_40(), params(1, 50, 0.05, 3.0, 3)},
  };
  for (const Case& c : cases) {
    INFO(c.name);
    SaParams all = c.base;
    all.replicas = 2 * kLanes + 5;
    all.first_replica = 7;
    SaParams first = c.base;
    first.replicas = 10;
    first.first_replica = 7;
    SaParams second = c.base;
    second.replicas = 2 * kLanes - 5;
    second.first_replica = 17;
    const SaResult whole = sbmc::run_sa_openmp(c.instance, all, 3);
    const SaResult a = sbmc::run_sa_openmp(c.instance, first, 3);
    const SaResult b = sbmc::run_sa_openmp(c.instance, second, 3);
    CHECK(b.first_replica == 17);
    std::vector<sbmc::SaReplicaResult> per_replica = a.per_replica;
    per_replica.insert(per_replica.end(), b.per_replica.begin(), b.per_replica.end());
    std::vector<Spin> best = a.best_spins;
    best.insert(best.end(), b.best_spins.begin(), b.best_spins.end());
    std::vector<Spin> final_spins = a.final_spins;
    final_spins.insert(final_spins.end(), b.final_spins.begin(), b.final_spins.end());
    CHECK(same_per_replica(per_replica, whole.per_replica));
    CHECK(best == whole.best_spins);
    CHECK(final_spins == whole.final_spins);
  }
}

TEST_CASE("C2: the last global replica indices") {
  const Graph four = rudy("four_nodes.rudy");
  const int replicas = 2 * kLanes + 5;
  const SaParams last = params(replicas, 10, 0.1, 3.0, 2, (std::int64_t{1} << 32) - replicas);
  SaResult c2;
  CHECK_NOTHROW(c2 = sbmc::run_sa_openmp(four, last));
  const SaResult c0 = sbmc::run_sa_sequential(four, last);
  const std::string differ = field_differences(c0, c2);
  INFO("fields that differ: " << shown(differ));
  CHECK(differ.empty());

  SaParams beyond = last;
  beyond.replicas = replicas + 1;  // the last global index would be 2^32
  CHECK_THROWS_AS(sbmc::run_sa_openmp(four, beyond), InvalidArgument);
}

TEST_CASE("C2: other properties") {
  CHECK((kLanes == 8 || kLanes == 16 || kLanes == 32 || kLanes == 64));
  const SaResult sequential =
      sbmc::run_sa_sequential(rudy("four_nodes.rudy"), params(3, 4, 0.1, 3.0, 1));
  CHECK(sequential.threads == 1);
  // The instances of the test, as specified.
  const Graph dense40 = dense_graph_40();
  CHECK(dense40.edges.size() == 689u);
  CHECK(dense40.edges.front().w == 2);
  CHECK(dense_graph_40_heavy().edges.front().w == 200);
  CHECK(heavy_torus().edges.size() == 48u);
  CHECK(huge_torus().edges.front().w == 500000);
}

TEST_CASE("C2: invalid arguments throw InvalidArgument") {
  const Graph four = rudy("four_nodes.rudy");
  const SaParams good = params(2, 3, 0.1, 3.0, 1);
  CHECK_NOTHROW(sbmc::run_sa_openmp(four, good, 0));

  // threads < 0, with the prefix of the function.
  const std::string threads_message = message_of([&] { sbmc::run_sa_openmp(four, good, -1); });
  INFO("threads = -1: '" << threads_message << "'");
  CHECK(threads_message.starts_with("run_sa_openmp: "));

  // The cases of "SA: invalid arguments throw InvalidArgument".
  const auto with = [&good](auto change) {
    SaParams p = good;
    change(p);
    return p;
  };
  const std::vector<SaParams> invalid = {
      with([](SaParams& p) { p.replicas = 0; }),
      with([](SaParams& p) { p.sweeps = 0; }),
      with([](SaParams& p) { p.beta0 = 0.0; }),
      with([](SaParams& p) { p.beta0 = -1.0; }),
      with([](SaParams& p) { p.beta0 = kNaN; }),
      with([](SaParams& p) { p.beta0 = kInf; }),
      with([](SaParams& p) { p.beta1 = kNaN; }),
      with([](SaParams& p) { p.beta1 = 0.05; }),  // beta1 < beta0
      with([](SaParams& p) { p.first_replica = -1; }),
      with([](SaParams& p) {
        p.sweeps = 1;
        p.replicas = 3;
        p.first_replica = (std::int64_t{1} << 32) - 2;  // the last index would be 2^32
      }),
  };
  for (std::size_t k = 0; k < invalid.size(); ++k) {
    INFO("invalid parameters, case " << k);
    const std::string message = message_of([&] { sbmc::run_sa_openmp(four, invalid[k], 2); });
    CHECK(message.starts_with("run_sa_openmp: "));
  }
  CHECK(message_of([&] { sbmc::run_sa_openmp(four, invalid[0]); }) ==
        "run_sa_openmp: replicas = 0 must be at least 1");

  // n > 32768 for a DenseSpinGlass: rejected at once, before total_weight (O(n^2)).
  CHECK(message_of([] {
          sbmc::run_sa_openmp(DenseSpinGlass{32769, 1}, params(1, 1, 0.1, 3.0, 1));
        }).starts_with("run_sa_openmp: n = 32769"));
  // An invalid instance: the error of total_weight.
  const Graph repeated{3, {{0, 1, 1}, {0, 1, 1}}};
  CHECK(message_of([&] { sbmc::run_sa_openmp(repeated, good); }).starts_with("total_weight"));
  // threads is checked first.
  CHECK(message_of([&] {
          sbmc::run_sa_openmp(four, invalid[0], -1);
        }).starts_with("run_sa_openmp: threads = -1"));
}
