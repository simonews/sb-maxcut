// Exact enumeration C3 (sbmc/bruteforce.hpp): reference optima, independence from the number of
// threads and the preconditions.
//
// The reference values (minimum energy, number of optimal partitions, canonical optimum) were
// computed by cldArchitect with an independent Gray-code enumeration in C++, checked against an
// exhaustive enumeration in NumPy on every instance up to 24 nodes. They are integers and hold in
// every preset.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <sbmc/bruteforce.hpp>
#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/rudy.hpp>

#ifndef SBMC_TEST_DATA_DIR
#error "SBMC_TEST_DATA_DIR must be defined by CMake (absolute path of tests/data)"
#endif

namespace {

using sbmc::BruteForceResult;
using sbmc::DenseSpinGlass;
using sbmc::Graph;
using sbmc::Instance;
using sbmc::InvalidArgument;
using sbmc::Spin;

Graph rudy(const char* name) {
  return sbmc::read_rudy(std::string(SBMC_TEST_DATA_DIR) + "/" + name);
}

// The configuration as a string of '+' and '-', spin 0 first.
std::string config_text(const std::vector<Spin>& spins) {
  std::string text;
  for (const Spin s : spins) {
    text.push_back(s == 1 ? '+' : (s == -1 ? '-' : '?'));
  }
  return text;
}

// Restores the OpenMP settings changed by a test, also when a check throws (as in
// test_sb_sequential.cpp).
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

struct ReferenceRow {
  std::string name;
  Instance instance;
  std::int64_t n;
  std::int64_t total_weight;
  std::int64_t min_energy;
  std::int64_t max_cut;
  std::uint64_t optimal_count;
  std::string config;
};

std::vector<ReferenceRow> reference_rows() {
  return {
      {"four_nodes.rudy", rudy("four_nodes.rudy"), 4, 5, -3, 4, 1, "+-+-"},
      {"torus_4x6_unit.rudy", rudy("torus_4x6_unit.rudy"), 24, 48, -48, 48, 1,
       "+-+-+--+-+-++-+-+--+-+-+"},
      {"torus_4x6_pm1.rudy", rudy("torus_4x6_pm1.rudy"), 24, 16, -32, 24, 35,
       "++-++--+--+-+--+--++++++"},
      {"sparse_24_unit.rudy", rudy("sparse_24_unit.rudy"), 24, 48, -32, 40, 1,
       "+++--+--+++++-----+--+--"},
      {"weighted_isolated_13.rudy", rudy("weighted_isolated_13.rudy"), 13, 2147483669, -10737420359,
       6442452014, 4, "+-+--+-++-+++"},
      {"DenseSpinGlass{16, 16}", DenseSpinGlass{16, 16}, 16, 12, -46, 29, 1, "++--+-+++---+-++"},
      {"DenseSpinGlass{20, 20}", DenseSpinGlass{20, 20}, 20, 10, -60, 35, 2,
       "++--++++-------+-+-+"},
      {"DenseSpinGlass{22, 22}", DenseSpinGlass{22, 22}, 22, -3, -71, 34, 1,
       "+-+--------++--+-+++--"},
      {"DenseSpinGlass{24, 24}", DenseSpinGlass{24, 24}, 24, 10, -84, 47, 1,
       "++++---+-+++-+-+-+----++"},
      {"Graph n = 1, no edges", Graph{1, {}}, 1, 0, 0, 0, 1, "+"},
      {"Graph n = 2, edge (0, 1, 5)", Graph{2, {{0, 1, 5}}}, 2, 5, -5, 5, 1, "+-"},
      {"Graph n = 2, edge (0, 1, -5)", Graph{2, {{0, 1, -5}}}, 2, -5, -5, 0, 1, "++"},
      {"Graph n = 3, no edges", Graph{3, {}}, 3, 0, 0, 0, 4, "+++"},
      {"DenseSpinGlass{1, 7}", DenseSpinGlass{1, 7}, 1, 0, 0, 0, 1, "+"},
      {"DenseSpinGlass{2, 0}", DenseSpinGlass{2, 0}, 2, -1, -1, 0, 1, "++"},
  };
}

// Every field of two results except threads.
bool same_solution(const BruteForceResult& a, const BruteForceResult& b) {
  return a.n == b.n && a.total_weight == b.total_weight && a.min_energy == b.min_energy &&
         a.max_cut == b.max_cut && a.optimal_count == b.optimal_count &&
         a.best_spins == b.best_spins;
}

}  // namespace

TEST_CASE("C3: reference optima") {
  for (const ReferenceRow& row : reference_rows()) {
    INFO(row.name);
    const BruteForceResult result = sbmc::solve_bruteforce(row.instance);
    CHECK(result.n == row.n);
    CHECK(result.total_weight == row.total_weight);
    CHECK(result.min_energy == row.min_energy);
    CHECK(result.max_cut == row.max_cut);
    CHECK(result.optimal_count == row.optimal_count);
    CHECK(config_text(result.best_spins) == row.config);

    REQUIRE(result.best_spins.size() == static_cast<std::size_t>(row.n));
    CHECK(result.best_spins[0] == 1);
    CHECK(sbmc::cut_direct(row.instance, result.best_spins) == result.max_cut);
    CHECK(sbmc::ising_energy(row.instance, result.best_spins) == result.min_energy);
    CHECK(result.threads >= 1);
  }
}

TEST_CASE("C3: the result does not depend on the number of threads") {
  const OmpSettingsGuard guard;
  omp_set_dynamic(0);  // the runtime must give every parallel region the requested team
  const int initial_threads = guard.initial_max_threads();
  struct Case {
    std::string name;
    Instance instance;
  };
  // torus_4x6_pm1 has 35 optimal partitions: the counts of the intervals are summed.
  const std::vector<Case> cases = {
      {"torus_4x6_pm1.rudy", rudy("torus_4x6_pm1.rudy")},
      {"weighted_isolated_13.rudy", rudy("weighted_isolated_13.rudy")},
      {"DenseSpinGlass{24, 24}", DenseSpinGlass{24, 24}},
  };
  const std::vector<int> requested = {1, 2, 3, 0, initial_threads};
  const std::vector<int> expected_team = {1, 2, 3, initial_threads, initial_threads};
  for (const Case& c : cases) {
    std::vector<BruteForceResult> results;
    for (const int threads : requested) {
      results.push_back(sbmc::solve_bruteforce(c.instance, threads));
      // The library changes no OpenMP setting (the number of threads goes into num_threads).
      CHECK(omp_get_max_threads() == initial_threads);
      CHECK(omp_get_dynamic() == 0);
    }
    for (std::size_t k = 0; k < results.size(); ++k) {
      INFO(c.name << ", threads = " << requested[k] << ": team of " << results[k].threads
                  << ", optimal_count " << results[k].optimal_count << ", min_energy "
                  << results[k].min_energy << ", config " << config_text(results[k].best_spins));
      CHECK(results[k].threads == expected_team[k]);
      CHECK(same_solution(results[k], results[0]));
    }
  }
  // Back to the initial settings.
  omp_set_dynamic(guard.initial_dynamic());
  omp_set_num_threads(initial_threads);
  CHECK(omp_get_max_threads() == initial_threads);
  CHECK(omp_get_dynamic() == guard.initial_dynamic());
}

TEST_CASE("C3: invalid arguments throw InvalidArgument") {
  const Instance four = rudy("four_nodes.rudy");
  CHECK_NOTHROW(sbmc::solve_bruteforce(four, 0));
  CHECK_THROWS_AS(sbmc::solve_bruteforce(four, -1), InvalidArgument);
  CHECK_THROWS_AS(sbmc::solve_bruteforce(DenseSpinGlass{41, 1}), InvalidArgument);
  CHECK_THROWS_AS(sbmc::solve_bruteforce(Graph{41, {}}), InvalidArgument);
  // n is checked before total_weight, which would take O(n^2) time here: with the wrong order the
  // case would not end.
  CHECK_THROWS_AS(sbmc::solve_bruteforce(DenseSpinGlass{2147483647, 1}), InvalidArgument);
  CHECK_THROWS_AS(sbmc::solve_bruteforce(DenseSpinGlass{0, 1}), InvalidArgument);
  CHECK_THROWS_AS(sbmc::solve_bruteforce(Graph{3, {{1, 2, 1}, {0, 1, 1}}}),
                  InvalidArgument);  // edges out of order
  CHECK_THROWS_AS(sbmc::solve_bruteforce(Graph{3, {{0, 3, 1}}}),
                  InvalidArgument);  // index outside [0, n)
}
